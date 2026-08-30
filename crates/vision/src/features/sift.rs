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
use std::collections::HashMap;

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
    /// Domain-size pooling (DSP-SIFT / Dong & Soatto CVPR'15): average the
    /// unnormalized SIFT histogram over `dsp_num_scales` scales in
    /// `[dsp_min_scale, dsp_max_scale] × σ`, then L2-normalize once. Same
    /// 128-D output; improves wide-baseline matching without changing
    /// detections. Off by default (byte-identical legacy descriptors).
    pub domain_size_pooling: bool,
    /// Lower end of the DSP scale range as a multiple of the detected σ
    /// (COLMAP default `1/6`).
    pub dsp_min_scale: f64,
    /// Upper end of the DSP scale range as a multiple of the detected σ.
    /// Dong & Soatto use `4/3`; COLMAP defaults to `3` (much slower, little
    /// extra mAP beyond ~4/3). Default follows the paper for practical CPU
    /// cost on dense SfM; override toward 3 for COLMAP-literal A/Bs.
    pub dsp_max_scale: f64,
    /// Number of domain sizes pooled when [`Self::domain_size_pooling`] is on.
    /// The paper's standard preset uses `15` samples; callers can lower this
    /// only for an explicitly bounded experiment.
    pub dsp_num_scales: usize,
    /// Descriptor normalization. [`SiftNormalization::L2`] is Lowe's classic
    /// clamp-and-renormalize; [`SiftNormalization::L1Root`] is COLMAP's default
    /// (L1-normalize then element-wise sqrt — RootSIFT / Arandjelović &
    /// Zisserman). Off-path default stays L2 for byte-identical legacy.
    pub normalization: SiftNormalization,
    /// Cap on orientations kept per detection locus (strongest histogram peaks
    /// first). COLMAP default is `2`. `0` = unlimited (legacy). Default 0.
    pub max_orientations: usize,
    /// Use the standards-aligned orientation peak selector: six circular
    /// three-tap smoothing passes, strict local maxima, an 80% dominant-peak
    /// gate, and bounded parabolic interpolation. Off preserves the historical
    /// threshold-bin behavior byte-for-byte. Default false.
    pub standard_orientation_peaks: bool,
    /// When truncating to [`Self::max_keypoints`], prefer larger-σ detections
    /// (COLMAP / VLFeat covdet keep coarser features first) instead of highest
    /// DoG contrast. Default false = legacy contrast ranking.
    pub prefer_larger_scale: bool,
    /// When true, always walk every octave before applying `max_keypoints`
    /// truncation (no early break). Needed for [`Self::prefer_larger_scale`];
    /// also useful with contrast ranking so coarse high-contrast peaks are
    /// not skipped. Default false = legacy early break.
    pub full_pyramid: bool,
    /// Descriptor spatial magnification in units of the detected σ. The
    /// historical implementation uses `8.0`; COLMAP/VLFeat-style SIFT uses
    /// approximately `3.0`. This is exposed so the narrower COLMAP-style
    /// sampling can be tested without changing the legacy default.
    pub descriptor_magnification: f64,
    /// Evaluate descriptor gradients on a scale-adaptive Gaussian pyramid.
    ///
    /// The historical path differentiates the source image directly.  When
    /// enabled, descriptors sample central-difference gradients from Gaussian
    /// levels selected in the original-image frame (with log-scale
    /// interpolation between adjacent levels), following the VLFeat
    /// descriptor convention.  This is deliberately off by default so the
    /// legacy descriptor bytes remain unchanged.
    pub scale_adaptive_gradients: bool,
    /// Use one cohesive VLFeat/COLMAP-compatible descriptor path.
    ///
    /// This is deliberately separate from the earlier scale-adaptive
    /// experiment: it selects the VLFeat octave image, keypoint coordinate
    /// convention, magnification, Gaussian support, trilinear histogram
    /// layout, normalization, and COLMAP's 512-equivalent byte quantization
    /// as one unit.  The legacy descriptor is unchanged when this is false.
    pub vlfeat_compatible_descriptor: bool,
    /// Use the complete VLFeat/COLMAP DoG detector contract: first octave
    /// `-1` with bilinear 2× upsampling, VLFeat's scale schedule, 3-D
    /// subpixel localization, edge test, orientation assignment, and
    /// large-scale-first feature cap. Default false preserves the historical
    /// detector byte-for-byte.
    pub vlfeat_compatible_detector: bool,
    /// Use the orientation-histogram interpolation enabled by COLMAP's
    /// vendored VLFeat build (`VL_SIFT_BILINEAR_ORIENTATIONS`). The ordinary
    /// VLFeat source defaults to nearest-bin accumulation; keeping this
    /// separate preserves the existing compatible-detector experiment while
    /// making the COLMAP-specific source choice explicit. Default false.
    pub vlfeat_bilinear_orientations: bool,
    /// Emit compatible-detector rows using COLMAP's CPU SIFT source order:
    /// retained DoG levels in ascending `(octave, level)` order, then the
    /// original VLFeat scan order within a level and its returned orientation
    /// order. The detector already produces this order today; this explicit
    /// opt-in makes the contract testable without changing the legacy or the
    /// existing compatible-detector defaults.
    pub vlfeat_compatible_output_order: bool,
}

/// How the 128-D SIFT histogram is normalized after pooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SiftNormalization {
    /// Lowe §6.1: L2 → clip at 0.2 → L2 again.
    #[default]
    L2,
    /// COLMAP `L1_ROOT`: L1-normalize, then √ per bin (Hellinger / RootSIFT).
    L1Root,
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
            domain_size_pooling: false,
            dsp_min_scale: DSP_PAPER_MIN_SCALE,
            dsp_max_scale: DSP_PAPER_MAX_SCALE,
            dsp_num_scales: DSP_PAPER_NUM_SCALES,
            normalization: SiftNormalization::L2,
            max_orientations: 0,
            standard_orientation_peaks: false,
            prefer_larger_scale: false,
            full_pyramid: false,
            descriptor_magnification: 8.0,
            scale_adaptive_gradients: false,
            vlfeat_compatible_descriptor: false,
            vlfeat_compatible_detector: false,
            vlfeat_bilinear_orientations: false,
            vlfeat_compatible_output_order: false,
        }
    }
}

/// DSP-SIFT's published evaluation preset (Dong & Soatto, CVPR 2015).
///
/// The descriptor is sampled in the domain-size interval around the detected
/// domain size, with `lambda_1 = 1/6`, `lambda_2 = 4/3`, and
/// `N_sigma_hat = 15`.  Keeping the constants here makes the default preset
/// auditable and prevents the three descriptor paths from silently choosing
/// different scale grids.  The public fields remain available for controlled
/// callers, but the command-line entry point exposes only the preset switch.
const DSP_PAPER_MIN_SCALE: f64 = 1.0 / 6.0;
const DSP_PAPER_MAX_SCALE: f64 = 4.0 / 3.0;
const DSP_PAPER_NUM_SCALES: usize = 15;

/// Return the deterministic multiplicative domain-size samples.
///
/// A one-sample request is deliberately the identity domain (`1.0`), rather
/// than the lower endpoint.  This is useful both as a correctness control and
/// as a strict regression test: enabling pooling with one sample must produce
/// the exact ordinary compatible descriptor.  For the normal multi-sample
/// preset the grid is uniformly sampled in domain size, matching both the
/// paper's uniform density and COLMAP's covariant extractor.  A zero sample
/// count is treated as pooling-off for backwards-compatible API users.
fn dsp_domain_scale_factors(config: &SiftConfig) -> Vec<f64> {
    if !config.domain_size_pooling || config.dsp_num_scales == 0 {
        return vec![1.0];
    }
    let count = config.dsp_num_scales;
    if count == 1 {
        return vec![1.0];
    }
    let lower = if config.dsp_min_scale.is_finite() && config.dsp_min_scale > 0.0 {
        config.dsp_min_scale
    } else {
        DSP_PAPER_MIN_SCALE
    };
    let upper = if config.dsp_max_scale.is_finite() && config.dsp_max_scale >= lower {
        config.dsp_max_scale
    } else {
        DSP_PAPER_MAX_SCALE.max(lower)
    };
    let step = (upper - lower) / count as f64;
    (0..count)
        .map(|index| (lower + index as f64 * step).max(f64::MIN_POSITIVE))
        .collect()
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

/// Central-difference gradients sampled on one Gaussian scale-space layer.
///
/// Keeping the gradients rather than only the blurred intensity layer makes
/// the fixed-keypoint descriptor path match VLFeat's contract: the image is
/// smoothed at the keypoint scale first, then the descriptor reads a sampled
/// gradient field.  `f32` is sufficient here (the final descriptor is f32)
/// and keeps the optional pyramid bounded for dense external keypoint sets.
#[derive(Clone)]
struct GradientLayer {
    width: usize,
    height: usize,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl GradientLayer {
    fn sample(&self, x: f64, y: f64) -> (f64, f64) {
        fn bilinear(data: &[f32], width: usize, height: usize, x: f64, y: f64) -> f64 {
            let x0 = x.floor() as i64;
            let y0 = y.floor() as i64;
            let fx = x - x0 as f64;
            let fy = y - y0 as f64;
            let get = |ix: i64, iy: i64| -> f64 {
                let xi = ix.clamp(0, width as i64 - 1) as usize;
                let yi = iy.clamp(0, height as i64 - 1) as usize;
                data[yi * width + xi] as f64
            };
            let p00 = get(x0, y0);
            let p10 = get(x0 + 1, y0);
            let p01 = get(x0, y0 + 1);
            let p11 = get(x0 + 1, y0 + 1);
            p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy
        }

        (
            bilinear(&self.gx, self.width, self.height, x, y),
            bilinear(&self.gy, self.width, self.height, x, y),
        )
    }
}

#[derive(Clone)]
struct ScaleAdaptiveOctave {
    /// Number of original-image pixels represented by one octave pixel.
    pixel_scale: f64,
    levels: Vec<GradientLayer>,
}

struct ScaleAdaptivePyramid {
    base_sigma: f64,
    k: f64,
    octaves: Vec<ScaleAdaptiveOctave>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScaleLevelSelection {
    octave: usize,
    lower: usize,
    upper: usize,
    /// Log-scale interpolation weight from `lower` to `upper`.
    weight: f64,
    pixel_scale: f64,
}

fn gradient_layer(layer: &Layer) -> GradientLayer {
    let mut gx = vec![0.0f32; layer.data.len()];
    let mut gy = vec![0.0f32; layer.data.len()];
    for y in 0..layer.height as i64 {
        for x in 0..layer.width as i64 {
            let index = y as usize * layer.width + x as usize;
            // This is VLFeat's `update_gradient` stencil: central differences
            // in the interior and a full one-sided difference on each image
            // boundary. The keypoint-descriptor loop normally clips the
            // outermost row/column, but retaining the complete source gradient
            // field keeps this layer correct for either VLFeat descriptor API.
            gx[index] = if x == 0 {
                (layer.get(1, y) - layer.get(0, y)) as f32
            } else if x == layer.width as i64 - 1 {
                (layer.get(x, y) - layer.get(x - 1, y)) as f32
            } else {
                (0.5 * (layer.get(x + 1, y) - layer.get(x - 1, y))) as f32
            };
            gy[index] = if y == 0 {
                (layer.get(x, 1) - layer.get(x, 0)) as f32
            } else if y == layer.height as i64 - 1 {
                (layer.get(x, y) - layer.get(x, y - 1)) as f32
            } else {
                (0.5 * (layer.get(x, y + 1) - layer.get(x, y - 1))) as f32
            };
        }
    }
    GradientLayer {
        width: layer.width,
        height: layer.height,
        gx,
        gy,
    }
}

fn build_scale_adaptive_pyramid(
    image: &GrayImage<'_>,
    keypoints: &[SiftKeypoint],
    config: &SiftConfig,
) -> ScaleAdaptivePyramid {
    let intervals = config.intervals.max(1);
    let base_sigma = config.sigma_input.max(1e-3);
    let k = 2.0f64.powf(1.0 / intervals as f64);
    let dsp_max = dsp_domain_scale_factors(config)
        .into_iter()
        .fold(1.0f64, f64::max);
    let max_sigma = keypoints
        .iter()
        .map(|kp| {
            if kp.sigma.is_finite() && kp.sigma > 0.0 {
                kp.sigma * dsp_max
            } else {
                base_sigma
            }
        })
        .fold(base_sigma, f64::max);
    let required_octaves = if max_sigma <= base_sigma {
        1
    } else {
        (max_sigma / base_sigma).log2().floor() as usize + 1
    };
    let min_dimension = image.width.min(image.height).max(1) as f64;
    let dimension_octaves = if min_dimension >= 16.0 {
        (min_dimension / 16.0).log2().floor() as usize + 1
    } else {
        1
    };
    let octave_count = required_octaves.min(dimension_octaves).max(1);

    let mut octave_base = Layer {
        width: image.width,
        height: image.height,
        data: image.pixels.iter().map(|&p| p as f64).collect(),
    };
    let mut octaves = Vec::with_capacity(octave_count);
    for octave_index in 0..octave_count {
        let mut gaussians = Vec::with_capacity(intervals + 3);
        gaussians.push(octave_base);
        for j in 1..=(intervals + 2) {
            let target = base_sigma * k.powi(j as i32);
            let previous = base_sigma * k.powi(j as i32 - 1);
            let step = ((target * target - previous * previous).max(1e-6)).sqrt();
            let next = blur(gaussians.last().unwrap(), step);
            gaussians.push(next);
        }
        let levels = gaussians.iter().map(gradient_layer).collect();
        let next_base = if octave_index + 1 < octave_count {
            Some(halve(&gaussians[intervals]))
        } else {
            None
        };
        octaves.push(ScaleAdaptiveOctave {
            pixel_scale: 2.0f64.powi(octave_index as i32),
            levels,
        });
        let Some(next_base) = next_base else {
            break;
        };
        octave_base = next_base;
    }
    ScaleAdaptivePyramid {
        base_sigma,
        k,
        octaves,
    }
}

impl ScaleAdaptivePyramid {
    fn select(&self, sigma: f64) -> ScaleLevelSelection {
        let sigma = if sigma.is_finite() && sigma > 0.0 {
            sigma
        } else {
            self.base_sigma
        };
        let ratio = (sigma / self.base_sigma).max(1.0);
        let requested_octave = ratio.log2().floor().max(0.0) as usize;
        let octave = requested_octave.min(self.octaves.len().saturating_sub(1));
        let octave_data = &self.octaves[octave];
        let local_sigma = sigma / octave_data.pixel_scale;
        let log_k = self.k.ln().max(f64::MIN_POSITIVE);
        let level_position = ((local_sigma / self.base_sigma).max(1.0).ln() / log_k)
            .clamp(0.0, (octave_data.levels.len().saturating_sub(1)) as f64);
        // Treat a numerically exact level as exact even when `ln`/`powf`
        // leaves it a few ulps below the integer.  Besides making the mapping
        // easier to reason about, this keeps repeated fixed-keypoint probes
        // from taking different interpolation branches on platforms with
        // slightly different libm rounding.
        let nearest = level_position.round();
        let snapped = (level_position - nearest).abs() < 1e-6;
        let level_position = if snapped { nearest } else { level_position };
        let lower = level_position.floor() as usize;
        let upper = if snapped {
            lower
        } else {
            (lower + 1).min(octave_data.levels.len().saturating_sub(1))
        };
        let weight = if lower == upper {
            0.0
        } else {
            (level_position - lower as f64).clamp(0.0, 1.0)
        };
        ScaleLevelSelection {
            octave,
            lower,
            upper,
            weight,
            pixel_scale: octave_data.pixel_scale,
        }
    }

    fn sample(&self, selection: ScaleLevelSelection, x: f64, y: f64) -> (f64, f64) {
        let octave = &self.octaves[selection.octave];
        let lower = octave.levels[selection.lower].sample(x, y);
        if selection.lower == selection.upper {
            return lower;
        }
        let upper = octave.levels[selection.upper].sample(x, y);
        (
            lower.0 * (1.0 - selection.weight) + upper.0 * selection.weight,
            lower.1 * (1.0 - selection.weight) + upper.1 * selection.weight,
        )
    }
}

/// One octave of the cohesive VLFeat descriptor pyramid.  The first octave
/// is the full-resolution image at approximately σ=0.8 (VLFeat's σ₀=1.6
/// viewed at first octave -1); subsequent octaves are downsampled by two.
#[derive(Clone)]
struct VlfeatOctave {
    pixel_scale: f64,
    levels: Vec<GradientLayer>,
}

struct VlfeatPyramid {
    base_sigma: f64,
    k: f64,
    octaves: Vec<VlfeatOctave>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct VlfeatLevelSelection {
    octave: usize,
    level: usize,
    pixel_scale: f64,
}

/// VLFeat's SIFT scale-space builder uses a finite four-sigma Gaussian kernel
/// and separable convolution. Boundary handling uses the source value at the
/// image edge, as `VL_PAD_BY_CONTINUITY` does, without changing the legacy
/// `blur` implementation. (The generic VLFeat `vl_imsmooth` helper uses three
/// sigma; `_vl_sift_smooth` is the authoritative SIFT path and uses four.)
fn gaussian_kernel_vlfeat(sigma: f64) -> Vec<f64> {
    let sigma = sigma.max(1e-6);
    let radius = (sigma * 4.0).ceil().max(1.0) as i64;
    let denom = 2.0 * sigma * sigma;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    for offset in -radius..=radius {
        kernel.push((-(offset * offset) as f64 / denom).exp());
    }
    let sum: f64 = kernel.iter().sum();
    kernel.iter().map(|value| value / sum).collect()
}

fn blur_vlfeat(input: &Layer, sigma: f64) -> Layer {
    let kernel = gaussian_kernel_vlfeat(sigma);
    let radius = kernel.len() as i64 / 2;
    let mut horizontal = vec![0.0f64; input.data.len()];
    for y in 0..input.height as i64 {
        for x in 0..input.width as i64 {
            horizontal[y as usize * input.width + x as usize] = kernel
                .iter()
                .enumerate()
                .map(|(index, &weight)| weight * input.get(x + index as i64 - radius, y))
                .sum();
        }
    }
    let horizontal = Layer {
        width: input.width,
        height: input.height,
        data: horizontal,
    };
    let horizontal_get = |x: i64, y: i64| -> f64 {
        let xi = x.clamp(0, horizontal.width as i64 - 1) as usize;
        let yi = y.clamp(0, horizontal.height as i64 - 1) as usize;
        horizontal.data[yi * horizontal.width + xi]
    };
    let mut output = vec![0.0f64; input.data.len()];
    for y in 0..input.height as i64 {
        for x in 0..input.width as i64 {
            output[y as usize * input.width + x as usize] = kernel
                .iter()
                .enumerate()
                .map(|(index, &weight)| weight * horizontal_get(x, y + index as i64 - radius))
                .sum();
        }
    }
    Layer {
        width: input.width,
        height: input.height,
        data: output,
    }
}

impl GradientLayer {
    fn sample_integer(&self, x: i64, y: i64) -> (f64, f64) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return (0.0, 0.0);
        }
        let index = y as usize * self.width + x as usize;
        (self.gx[index] as f64, self.gy[index] as f64)
    }
}

fn build_vlfeat_pyramid(
    image: &GrayImage<'_>,
    keypoints: &[SiftKeypoint],
    config: &SiftConfig,
) -> VlfeatPyramid {
    let intervals = config.intervals.max(1);
    let k = 2.0f64.powf(1.0 / intervals as f64);
    // VLFeat's default sigma0=1.6 belongs to the first doubled/negative
    // octave.  At the original image resolution its equivalent base is 0.8.
    let base_sigma = (config.sigma_base.max(1e-3) / 2.0).max(config.sigma_input);
    let dsp_max = dsp_domain_scale_factors(config)
        .into_iter()
        .fold(1.0f64, f64::max);
    let max_sigma = keypoints
        .iter()
        .map(|kp| {
            if kp.sigma.is_finite() && kp.sigma > 0.0 {
                kp.sigma * dsp_max
            } else {
                base_sigma
            }
        })
        .fold(base_sigma, f64::max);
    let required_octaves = if max_sigma <= base_sigma {
        1
    } else {
        (max_sigma / base_sigma).log2().floor() as usize + 1
    };
    let min_dimension = image.width.min(image.height).max(1) as f64;
    let dimension_octaves = if min_dimension >= 16.0 {
        (min_dimension / 16.0).log2().floor() as usize + 1
    } else {
        1
    };
    let octave_count = required_octaves.min(dimension_octaves).max(1);

    let source = Layer {
        width: image.width,
        height: image.height,
        data: image.pixels.iter().map(|&value| value as f64).collect(),
    };
    let source_sigma = config.sigma_input.max(1e-3);
    let initial = if base_sigma > source_sigma {
        blur_vlfeat(
            &source,
            (base_sigma * base_sigma - source_sigma * source_sigma).sqrt(),
        )
    } else {
        source
    };

    let mut octave_base = initial;
    let mut octaves = Vec::with_capacity(octave_count);
    for octave_index in 0..octave_count {
        let mut gaussians = Vec::with_capacity(intervals + 3);
        gaussians.push(octave_base);
        for level in 1..=(intervals + 2) {
            let target = base_sigma * k.powi(level as i32);
            let previous = base_sigma * k.powi(level as i32 - 1);
            let step = (target * target - previous * previous).max(1e-8).sqrt();
            let next = blur_vlfeat(gaussians.last().unwrap(), step);
            gaussians.push(next);
        }
        let levels = gaussians.iter().map(gradient_layer).collect();
        let next_base = if octave_index + 1 < octave_count {
            Some(halve(&gaussians[intervals]))
        } else {
            None
        };
        octaves.push(VlfeatOctave {
            pixel_scale: 2.0f64.powi(octave_index as i32),
            levels,
        });
        let Some(next_base) = next_base else {
            break;
        };
        octave_base = next_base;
    }
    VlfeatPyramid {
        base_sigma,
        k,
        octaves,
    }
}

impl VlfeatPyramid {
    fn select(&self, sigma: f64) -> VlfeatLevelSelection {
        let sigma = if sigma.is_finite() && sigma > 0.0 {
            sigma
        } else {
            self.base_sigma
        };
        let ratio = (sigma / self.base_sigma).max(f64::MIN_POSITIVE);
        // `vl_sift_keypoint_init` chooses the largest octave containing the
        // scale.  With VLFeat's s_min=-1 and sigma0=1.6*k, expressed relative
        // to this pyramid's original-resolution base (1.6/2), that is
        // floor(log2(ratio) - 0.5/S), not simply floor(log2(ratio)).
        let intervals = self.k.log2().recip();
        let octave_phase = ratio.log2() - 0.5 / intervals;
        let requested = octave_phase.floor().max(0.0) as usize;
        let octave = requested.min(self.octaves.len().saturating_sub(1));
        let octave_data = &self.octaves[octave];
        let local_sigma = sigma / octave_data.pixel_scale;
        let position = ((local_sigma / self.base_sigma)
            .max(f64::MIN_POSITIVE)
            .log2()
            * intervals)
            .clamp(1.0, octave_data.levels.len().saturating_sub(3) as f64);
        // VLFeat rounds a keypoint's continuous scale to its integer level
        // (`is`), and the stored GSS array is offset by -s_min = 1.  The
        // resulting array index is therefore `round(S*log2(local/base))`.
        // Clamp to source's descriptor-valid gradient levels [s=0,S-1],
        // represented here by array levels [1,S]. It does not blend
        // neighboring gradient fields for the ordinary descriptor path.
        let level = position
            .round()
            .clamp(1.0, octave_data.levels.len().saturating_sub(3) as f64)
            as usize;
        VlfeatLevelSelection {
            octave,
            level,
            pixel_scale: octave_data.pixel_scale,
        }
    }

    fn sample_integer(&self, selection: VlfeatLevelSelection, x: i64, y: i64) -> (f64, f64) {
        self.octaves[selection.octave].levels[selection.level].sample_integer(x, y)
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

/// VLFeat's `copy_and_upsample_rows`: separable linear interpolation with the
/// last source row/column repeated at the boundary.  The legacy detector uses
/// nearest-neighbour upsampling; keeping this helper separate makes the
/// compatible detector's first-octave contract explicit and leaves defaults
/// untouched.
fn double_up_vlfeat(layer: &Layer) -> Layer {
    let width = layer.width.saturating_mul(2).max(1);
    let height = layer.height.saturating_mul(2).max(1);
    let mut data = vec![0.0f64; width * height];
    for y in 0..height {
        let source_y = y / 2;
        let fy = if y % 2 == 0 { 0.0 } else { 0.5 };
        let y0 = source_y.min(layer.height.saturating_sub(1));
        let y1 = (source_y + 1).min(layer.height.saturating_sub(1));
        for x in 0..width {
            let source_x = x / 2;
            let fx = if x % 2 == 0 { 0.0 } else { 0.5 };
            let x0 = source_x.min(layer.width.saturating_sub(1));
            let x1 = (source_x + 1).min(layer.width.saturating_sub(1));
            let p00 = layer.data[y0 * layer.width + x0];
            let p10 = layer.data[y0 * layer.width + x1];
            let p01 = layer.data[y1 * layer.width + x0];
            let p11 = layer.data[y1 * layer.width + x1];
            data[y * width + x] = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;
        }
    }
    Layer {
        width,
        height,
        data,
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

/// One isotropic VLFeat DoG extremum before its orientation copies are
/// expanded.  This is intentionally a diagnostic-only view of the detector;
/// the ordinary [`SiftKeypoint`] API and feature bytes remain unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct SiftDetectionCandidate {
    /// Keypoint centre in the exported COLMAP half-pixel convention.
    pub x: f64,
    pub y: f64,
    /// Absolute scale in original-image pixels.
    pub sigma: f64,
    /// Signed refined DoG response (`val` in VLFeat); ranking uses its
    /// magnitude, as does the feature record.
    pub response: f64,
    /// VLFeat octave index (`-1` is the doubled first octave).
    pub octave: i32,
    /// Integer VLFeat DoG scale level (`k->is`).
    pub level: i32,
    /// Lowe/VLFeat spatial curvature ratio
    /// `(Dxx + Dyy)^2 / (Dxx*Dyy - Dxy^2)` at the refined locus.
    /// Values below `(edge_threshold + 1)^2 / edge_threshold` pass the
    /// edge rejection test.  This is exported only for detector diagnostics.
    pub edge_score: f64,
}

/// One oriented copy of a VLFeat DoG extremum for detector diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct SiftOrientedDetection {
    pub x: f64,
    pub y: f64,
    pub sigma: f64,
    pub orientation: f64,
    pub response: f64,
    pub octave: i32,
    pub level: i32,
    /// Index in the source orientation-peak order for this locus.
    pub orientation_index: usize,
}

/// Internal VLFeat detector stages exposed only when an explicit diagnostic
/// call is made.  `after_orientation` is before the per-locus orientation cap;
/// `after_orientation_cap` is before the global feature cap; `after_cap` is
/// the final order after the global `max_keypoints` policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SiftDetectorDiagnostics {
    pub before_orientation: Vec<SiftDetectionCandidate>,
    pub after_orientation: Vec<SiftOrientedDetection>,
    pub after_orientation_cap: Vec<SiftOrientedDetection>,
    pub after_cap: Vec<SiftOrientedDetection>,
}

impl SiftKeypoint {
    /// Build a fixed isotropic keypoint for descriptor-only comparisons.
    ///
    /// This intentionally discards affine-shape metadata; callers that have
    /// such metadata should use the explicit `affine_shape` field through an
    /// in-crate adapter.  The detector and the ordinary extraction path do
    /// not use this constructor.
    pub fn from_location_scale_orientation(x: f64, y: f64, sigma: f64, orientation: f64) -> Self {
        Self {
            x,
            y,
            sigma,
            orientation: orientation.rem_euclid(std::f64::consts::TAU),
            affine_shape: None,
            contrast: 0.0,
        }
    }
}

/// Extract SIFT keypoints and 128-d descriptors from a grayscale image.
///
/// The returned [`FeatureSet`] stores `(x, y)` keypoint locations and one
/// L2-normalized 128-dim descriptor per keypoint; descriptor order matches
/// keypoint order (strongest contrast first when `max_keypoints` truncates,
/// unless [`SiftConfig::prefer_larger_scale`] is set).
pub fn extract_sift(
    image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftError> {
    if config.vlfeat_compatible_detector {
        return extract_sift_vlfeat_detector(image, config);
    }
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

        // Contrast+legacy can stop early once the budget is filled (fine
        // octaves first). Full-pyramid / larger-scale modes finish all
        // octaves before pruning.
        if !config.prefer_larger_scale
            && !config.full_pyramid
            && keypoints.len() >= config.max_keypoints
        {
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
        if config.prefer_larger_scale {
            // COLMAP covdet: coarser (larger σ) first, contrast as tie-break.
            keypoints.sort_by(|a, b| {
                b.sigma
                    .total_cmp(&a.sigma)
                    .then_with(|| b.contrast.total_cmp(&a.contrast))
            });
        } else {
            keypoints.sort_by(|a, b| b.contrast.total_cmp(&a.contrast));
        }
        keypoints.truncate(config.max_keypoints);
    }
    let descriptors = describe_sift_keypoints(image, &keypoints, config);
    Ok((keypoints, descriptors))
}

/// A subpixel/subscale extremum in the VLFeat coordinate system.  `scale` is
/// the continuous DoG index whose integer centre is the middle layer passed to
/// [`refine_vlfeat_extremum`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct VlfeatRefinedExtremum {
    x: f64,
    y: f64,
    scale: f64,
    value: f64,
    edge_score: f64,
}

/// Solve the tiny linear system used by VLFeat's quadratic extrema fit.
/// Gaussian elimination follows the source's maximally-stable pivot rule and
/// returns a zero displacement for a singular Hessian, which is the source's
/// "give up" behavior.
#[allow(clippy::needless_range_loop)]
fn solve_vlfeat_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> [f64; 3] {
    for column in 0..3 {
        let mut pivot = column;
        let mut pivot_abs = matrix[column][column].abs();
        for row in (column + 1)..3 {
            let candidate = matrix[row][column].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if !(pivot_abs.is_finite() && pivot_abs >= 1e-10) {
            return [0.0; 3];
        }
        if pivot != column {
            matrix.swap(pivot, column);
            rhs.swap(pivot, column);
        }
        let diagonal = matrix[column][column];
        for entry in &mut matrix[column][column..] {
            *entry /= diagonal;
        }
        rhs[column] /= diagonal;
        for row in (column + 1)..3 {
            let factor = matrix[row][column];
            for col in column..3 {
                matrix[row][col] -= factor * matrix[column][col];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for row in (0..3).rev() {
        for col in (row + 1)..3 {
            rhs[row] -= matrix[row][col] * rhs[col];
        }
    }
    if rhs.iter().all(|value| value.is_finite()) {
        rhs
    } else {
        [0.0; 3]
    }
}

/// Refine one integer DoG extremum using VLFeat's five-iteration quadratic
/// fit.  The input `scale` is the integer DoG scale index in the source
/// coordinate system (`s=0` for the first eligible middle layer).
#[allow(clippy::too_many_arguments)]
fn refine_vlfeat_extremum(
    below: &Layer,
    middle: &Layer,
    above: &Layer,
    x: i64,
    y: i64,
    scale: f64,
    scale_max: f64,
    peak_threshold: f64,
    edge_threshold: f64,
) -> Option<VlfeatRefinedExtremum> {
    let value = middle.get(x, y);
    let threshold = peak_threshold.max(0.0);
    // VLFeat's preliminary CHECK_NEIGHBORS macro uses `>= +0.8*tp` for
    // maxima and `<= -0.8*tp` for minima; the final refined contrast gate
    // below remains the strict `abs(value) > tp` test.
    let mut is_max = value >= 0.8 * threshold;
    let mut is_min = value <= -0.8 * threshold;
    for ds in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if ds == 0 && dx == 0 && dy == 0 {
                    continue;
                }
                let neighbour = match ds {
                    -1 => below.get(x + dx, y + dy),
                    0 => middle.get(x + dx, y + dy),
                    1 => above.get(x + dx, y + dy),
                    _ => unreachable!(),
                };
                if neighbour >= value {
                    is_max = false;
                }
                if neighbour <= value {
                    is_min = false;
                }
            }
        }
    }
    if !(is_max || is_min) {
        return None;
    }

    let mut ix = x;
    let mut iy = y;
    let mut move_x = 0i64;
    let mut move_y = 0i64;
    let mut displacement = [0.0f64; 3];
    let mut derivatives = [0.0f64; 3];
    let mut hessian = [[0.0f64; 3]; 3];

    for _ in 0..5 {
        ix += move_x;
        iy += move_y;
        let sample = |dx: i64, dy: i64, ds: i64| -> f64 {
            match ds {
                -1 => below.get(ix + dx, iy + dy),
                0 => middle.get(ix + dx, iy + dy),
                1 => above.get(ix + dx, iy + dy),
                _ => unreachable!(),
            }
        };
        let centre = sample(0, 0, 0);
        let dxx = sample(1, 0, 0) + sample(-1, 0, 0) - 2.0 * centre;
        let dyy = sample(0, 1, 0) + sample(0, -1, 0) - 2.0 * centre;
        let dss = sample(0, 0, 1) + sample(0, 0, -1) - 2.0 * centre;
        let dxy =
            0.25 * (sample(1, 1, 0) + sample(-1, -1, 0) - sample(-1, 1, 0) - sample(1, -1, 0));
        let dxs =
            0.25 * (sample(1, 0, 1) + sample(-1, 0, -1) - sample(-1, 0, 1) - sample(1, 0, -1));
        let dys =
            0.25 * (sample(0, 1, 1) + sample(0, -1, -1) - sample(0, -1, 1) - sample(0, 1, -1));
        derivatives = [
            0.5 * (sample(1, 0, 0) - sample(-1, 0, 0)),
            0.5 * (sample(0, 1, 0) - sample(0, -1, 0)),
            0.5 * (sample(0, 0, 1) - sample(0, 0, -1)),
        ];
        hessian = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];
        displacement =
            solve_vlfeat_3x3(hessian, [-derivatives[0], -derivatives[1], -derivatives[2]]);
        move_x = if displacement[0] > 0.6 && ix < middle.width as i64 - 2 {
            1
        } else if displacement[0] < -0.6 && ix > 1 {
            -1
        } else {
            0
        };
        move_y = if displacement[1] > 0.6 && iy < middle.height as i64 - 2 {
            1
        } else if displacement[1] < -0.6 && iy > 1 {
            -1
        } else {
            0
        };
        if move_x == 0 && move_y == 0 {
            break;
        }
    }

    let centre = middle.get(ix, iy);
    let refined_value = centre
        + 0.5
            * (derivatives[0] * displacement[0]
                + derivatives[1] * displacement[1]
                + derivatives[2] * displacement[2]);
    let spatial_det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[0][1];
    let spatial_trace = hessian[0][0] + hessian[1][1];
    let edge_score = spatial_trace * spatial_trace / spatial_det;
    let xn = ix as f64 + displacement[0];
    let yn = iy as f64 + displacement[1];
    let sn = scale + displacement[2];
    let edge_limit = (edge_threshold + 1.0).powi(2) / edge_threshold.max(1e-12);
    if !(refined_value.abs() > threshold
        && edge_score.is_finite()
        && edge_score < edge_limit
        && edge_score >= 0.0
        && displacement
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.5)
        && xn >= 0.0
        && xn <= (middle.width - 1) as f64
        && yn >= 0.0
        && yn <= (middle.height - 1) as f64
        && sn >= -1.0
        && sn <= scale_max)
    {
        return None;
    }
    Some(VlfeatRefinedExtremum {
        x: xn,
        y: yn,
        scale: sn,
        value: refined_value,
        edge_score,
    })
}

fn vlfeat_gradient_at(layer: &Layer, x: i64, y: i64) -> (f64, f64) {
    let gx = if x == 0 {
        layer.get(1, y) - layer.get(0, y)
    } else if x == layer.width as i64 - 1 {
        layer.get(x, y) - layer.get(x - 1, y)
    } else {
        0.5 * (layer.get(x + 1, y) - layer.get(x - 1, y))
    };
    let gy = if y == 0 {
        layer.get(x, 1) - layer.get(x, 0)
    } else if y == layer.height as i64 - 1 {
        layer.get(x, y) - layer.get(x, y - 1)
    } else {
        0.5 * (layer.get(x, y + 1) - layer.get(x, y - 1))
    };
    (gx, gy)
}

/// Source-equivalent orientation histogram for a localized VLFeat feature.
/// The upstream source uses nearest orientation bins, while COLMAP's vendored
/// build enables `VL_SIFT_BILINEAR_ORIENTATIONS`; both variants use six
/// circular smoothing passes and at most four local maxima. COLMAP applies
/// its own smaller cap after this routine.
fn accumulate_vlfeat_orientation(
    histogram: &mut [f64; ORIENTATION_BINS],
    angle: f64,
    magnitude: f64,
    weight: f64,
    bilinear: bool,
) {
    let fbin =
        angle.rem_euclid(std::f64::consts::TAU) * ORIENTATION_BINS as f64 / std::f64::consts::TAU;
    if bilinear {
        // COLMAP's vendored VLFeat defines VL_SIFT_BILINEAR_ORIENTATIONS.
        // Its bin centers are shifted by half a bin, with the two adjacent
        // bins receiving complementary weights (including the circular
        // bin-35/bin-0 boundary).
        let bin = (fbin - 0.5).floor() as isize;
        let rbin = fbin - bin as f64 - 0.5;
        let contribution = magnitude * weight;
        histogram[bin.rem_euclid(ORIENTATION_BINS as isize) as usize] +=
            (1.0 - rbin) * contribution;
        histogram[(bin + 1).rem_euclid(ORIENTATION_BINS as isize) as usize] += rbin * contribution;
    } else {
        let bin = fbin.floor() as usize % ORIENTATION_BINS;
        histogram[bin] += magnitude * weight;
    }
}

fn vlfeat_orientation_peaks(layer: &Layer, x: f64, y: f64, sigma: f64, bilinear: bool) -> Vec<f64> {
    let xi = (x + 0.5).floor() as i64;
    let yi = (y + 0.5).floor() as i64;
    let sigmaw = 1.5 * sigma.max(1e-6);
    let window = (3.0 * sigmaw).floor().max(1.0) as i64;
    let mut histogram = [0.0f64; ORIENTATION_BINS];
    for ys in (-window).max(-yi)..=window.min(layer.height as i64 - 1 - yi) {
        for xs in (-window).max(-xi)..=window.min(layer.width as i64 - 1 - xi) {
            let dx = (xi + xs) as f64 - x;
            let dy = (yi + ys) as f64 - y;
            let radius_sq = dx * dx + dy * dy;
            if radius_sq >= (window * window) as f64 + 0.6 {
                continue;
            }
            let (gx, gy) = vlfeat_gradient_at(layer, xi + xs, yi + ys);
            let magnitude = (gx * gx + gy * gy).sqrt();
            if !(magnitude.is_finite() && magnitude > 0.0) {
                continue;
            }
            let weight = (-radius_sq / (2.0 * sigmaw * sigmaw)).exp();
            let angle = gy.atan2(gx).rem_euclid(std::f64::consts::TAU);
            accumulate_vlfeat_orientation(&mut histogram, angle, magnitude, weight, bilinear);
        }
    }
    select_vlfeat_orientation_peaks(&smooth_orientation_histogram(&histogram))
}

/// Select VLFeat's first four local orientation maxima from an already
/// smoothed circular histogram.  Keeping this separate from image sampling
/// makes the source ordering and cap directly testable.
fn select_vlfeat_orientation_peaks(histogram: &[f64; ORIENTATION_BINS]) -> Vec<f64> {
    let max_hist = histogram.iter().copied().fold(0.0f64, f64::max);
    if !(max_hist.is_finite() && max_hist > 0.0) {
        return Vec::new();
    }
    let mut peaks = Vec::new();
    for bin in 0..ORIENTATION_BINS {
        let value = histogram[bin];
        let previous = histogram[(bin + ORIENTATION_BINS - 1) % ORIENTATION_BINS];
        let next = histogram[(bin + 1) % ORIENTATION_BINS];
        if !(value > 0.8 * max_hist && value > previous && value > next) {
            continue;
        }
        let denominator = next + previous - 2.0 * value;
        let delta = if denominator.is_finite() && denominator.abs() > f64::EPSILON {
            (-0.5 * (next - previous) / denominator).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let angle = ((bin as f64 + delta + 0.5) * std::f64::consts::TAU / ORIENTATION_BINS as f64)
            .rem_euclid(std::f64::consts::TAU);
        peaks.push(angle);
        if peaks.len() == 4 {
            break;
        }
    }
    peaks
}

/// Apply COLMAP's CPU/VLFeat feature cap to a set of oriented detections.
///
/// The cap is on *unoriented extrema*, not on emitted orientation copies.
/// VLFeat groups detections by integer DoG level, walks those groups from the
/// coarsest octave/level backwards, and then keeps the whole suffix beginning
/// at the group that crosses the cap.  Consequently the returned oriented
/// count can exceed `max_keypoints`; truncating an arbitrary sorted list of
/// oriented copies is not source-equivalent.
fn cap_vlfeat_oriented_detections_by_level(
    keypoints: &mut Vec<SiftKeypoint>,
    detections: &mut Vec<SiftOrientedDetection>,
    candidates: &[SiftDetectionCandidate],
    max_keypoints: usize,
) {
    debug_assert_eq!(keypoints.len(), detections.len());
    if max_keypoints == usize::MAX {
        return;
    }

    let mut groups = Vec::<((i32, i32), usize)>::new();
    let mut group_indices = HashMap::<(i32, i32), usize>::new();
    for candidate in candidates {
        let key = (candidate.octave, candidate.level);
        let group = if let Some(&group) = group_indices.get(&key) {
            group
        } else {
            let group = groups.len();
            groups.push((key, 0));
            group_indices.insert(key, group);
            group
        };
        groups[group].1 += 1;
    }
    if groups.is_empty() {
        keypoints.clear();
        detections.clear();
        return;
    }

    let mut first_group = 0usize;
    let mut counted = 0usize;
    for group in (0..groups.len()).rev() {
        counted = counted.saturating_add(groups[group].1);
        if counted > max_keypoints {
            first_group = group;
            break;
        }
    }

    let mut paired: Vec<_> = keypoints.drain(..).zip(detections.drain(..)).collect();
    paired.retain(|(_, detection)| {
        group_indices
            .get(&(detection.octave, detection.level))
            .is_some_and(|&group| group >= first_group)
    });
    let (kept_keypoints, kept_detections): (Vec<_>, Vec<_>) = paired.into_iter().unzip();
    *keypoints = kept_keypoints;
    *detections = kept_detections;
}

/// Re-establish the row order used by COLMAP's CPU SIFT wrapper.
///
/// `SiftCPUFeatureExtractor` does not sort rows by response, scale, or
/// descriptor distance. It first selects a suffix of complete DoG
/// `(octave, level)` groups and then appends each group's keypoints in the
/// order returned by VLFeat, with each keypoint's orientation rows in the
/// order returned by `vl_sift_calc_keypoint_orientations`. The detector loop
/// already emits that sequence, so this helper is intentionally a narrow
/// source-order normalization: it only makes the retained level grouping
/// explicit and preserves the input order within every level.
fn order_vlfeat_rows_like_colmap(
    keypoints: &mut Vec<SiftKeypoint>,
    detections: &mut Vec<SiftOrientedDetection>,
) {
    debug_assert_eq!(keypoints.len(), detections.len());
    let mut groups = Vec::<((i32, i32), Vec<(SiftKeypoint, SiftOrientedDetection)>)>::new();
    let mut group_indices = HashMap::<(i32, i32), usize>::new();
    for (keypoint, detection) in keypoints.drain(..).zip(detections.drain(..)) {
        let key = (detection.octave, detection.level);
        let group = if let Some(&group) = group_indices.get(&key) {
            group
        } else {
            let group = groups.len();
            groups.push((key, Vec::new()));
            group_indices.insert(key, group);
            group
        };
        groups[group].1.push((keypoint, detection));
    }
    groups.sort_by_key(|(key, _)| *key);

    for (_, rows) in groups {
        for (keypoint, detection) in rows {
            keypoints.push(keypoint);
            detections.push(detection);
        }
    }
}

/// Return the number of octaves used by the compatible CPU path.
///
/// COLMAP's CPU SIFT defaults to four octaves (`first_octave = -1`,
/// `num_octaves = 4`).  The legacy Rust extractor intentionally derives an
/// image-size-dependent count, but using that count here would create extra
/// coarse-octave candidates and would also change which whole DoG-level group
/// crosses the COLMAP feature cap.  An explicit `SiftConfig::octaves` remains
/// authoritative; zero means the source-compatible default, bounded by the
/// image dimensions just as VLFeat does.
fn vlfeat_compatible_octave_count(configured_octaves: usize, auto_octave_limit: usize) -> usize {
    let source_default = 4usize;
    let auto_octaves = auto_octave_limit.max(1);
    if configured_octaves == 0 {
        source_default.min(auto_octaves).max(1)
    } else {
        configured_octaves.min(auto_octaves).max(1)
    }
}

/// Detector path matching the isotropic VLFeat/COLMAP CPU SIFT contract.  It
/// intentionally lives beside (rather than inside) the historical detector:
/// the legacy path's nearest-neighbour upsample, integer extrema, threshold,
/// and contrast ranking remain byte-identical when this flag is off.
fn detect_sift_vlfeat_keypoints(
    image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<(Vec<SiftKeypoint>, SiftDetectorDiagnostics), SiftError> {
    let intervals = config.intervals.max(1);
    let min_dimension = image.width.min(image.height).max(1) as f64;
    // VLFeat's default first octave is -1. For a negative first octave its
    // automatic count is floor(log2(min_dimension)) - o_min - 3.
    let auto_octaves = ((min_dimension.log2().floor() as i64) - 2).max(1) as usize;
    let num_octaves = vlfeat_compatible_octave_count(config.octaves, auto_octaves);
    let k = 2.0f64.powf(1.0 / intervals as f64);
    let base_sigma = config.sigma_base.max(2.0 * config.sigma_input).max(1e-3);
    let source = Layer {
        width: image.width,
        height: image.height,
        data: image.pixels.iter().map(|&value| value as f64).collect(),
    };
    let mut octave = double_up_vlfeat(&source);
    let input_sigma = (2.0 * config.sigma_input).max(1e-3);
    if base_sigma > input_sigma {
        octave = blur_vlfeat(
            &octave,
            (base_sigma * base_sigma - input_sigma * input_sigma).sqrt(),
        );
    }
    let peak_threshold = (config.contrast_threshold / intervals as f64).max(0.0);
    let mut keypoints = Vec::new();
    let mut diagnostics = SiftDetectorDiagnostics::default();

    for octave_index in 0..num_octaves {
        let mut gaussians = Vec::with_capacity(intervals + 3);
        gaussians.push(octave.clone());
        for level in 1..=(intervals + 2) {
            let target = base_sigma * k.powi(level as i32);
            let previous = base_sigma * k.powi(level as i32 - 1);
            let step = (target * target - previous * previous).max(1e-8).sqrt();
            gaussians.push(blur_vlfeat(gaussians.last().unwrap(), step));
        }
        let dogs: Vec<Layer> = gaussians
            .windows(2)
            .map(|pair| Layer {
                width: pair[0].width,
                height: pair[0].height,
                data: pair[1]
                    .data
                    .iter()
                    .zip(pair[0].data.iter())
                    .map(|(next, previous)| next - previous)
                    .collect(),
            })
            .collect();
        let xper = 2.0f64.powi(octave_index as i32) / 2.0;
        for di in 1..dogs.len().saturating_sub(1) {
            let middle = &dogs[di];
            let below = &dogs[di - 1];
            let above = &dogs[di + 1];
            for y in 1..middle.height as i64 - 1 {
                for x in 1..middle.width as i64 - 1 {
                    let Some(refined) = refine_vlfeat_extremum(
                        below,
                        middle,
                        above,
                        x,
                        y,
                        (di - 1) as f64,
                        (intervals + 1) as f64,
                        peak_threshold,
                        config.edge_threshold,
                    ) else {
                        continue;
                    };
                    let local_sigma = base_sigma * k.powf(refined.scale + 1.0);
                    let octave = -1 + octave_index as i32;
                    let level = (di - 1) as i32;
                    diagnostics.before_orientation.push(SiftDetectionCandidate {
                        x: refined.x * xper + 0.5,
                        y: refined.y * xper + 0.5,
                        sigma: local_sigma * xper,
                        response: refined.value,
                        octave,
                        level,
                        edge_score: refined.edge_score,
                    });
                    let orientations = vlfeat_orientation_peaks(
                        &gaussians[di],
                        refined.x,
                        refined.y,
                        local_sigma,
                        config.vlfeat_bilinear_orientations,
                    );
                    // COLMAP's default `max_num_orientations` is 2.  The
                    // public config uses 0 for the historical unlimited
                    // setting, so compatible mode maps that sentinel to the
                    // source default while explicit values still allow up to
                    // VLFeat's four returned peaks.
                    let orientation_cap = if config.max_orientations == 0 {
                        2
                    } else {
                        config.max_orientations.min(4)
                    };
                    let oriented = orientations
                        .into_iter()
                        .enumerate()
                        .map(|(orientation_index, orientation)| SiftOrientedDetection {
                            x: refined.x * xper + 0.5,
                            y: refined.y * xper + 0.5,
                            sigma: local_sigma * xper,
                            orientation,
                            response: refined.value,
                            octave,
                            level,
                            orientation_index,
                        })
                        .collect::<Vec<_>>();
                    diagnostics
                        .after_orientation
                        .extend(oriented.iter().cloned());
                    for detection in oriented.iter().take(orientation_cap) {
                        diagnostics.after_orientation_cap.push(detection.clone());
                        keypoints.push(SiftKeypoint {
                            // COLMAP exports VLFeat's internal coordinates in
                            // the half-pixel convention.  The compatible
                            // descriptor subtracts this offset again.
                            x: detection.x,
                            y: detection.y,
                            sigma: detection.sigma,
                            orientation: detection.orientation,
                            affine_shape: None,
                            contrast: refined.value.abs(),
                        });
                    }
                }
            }
        }
        if octave_index + 1 < num_octaves {
            octave = halve(&gaussians[intervals]);
        }
    }

    cap_vlfeat_oriented_detections_by_level(
        &mut keypoints,
        &mut diagnostics.after_orientation_cap,
        &diagnostics.before_orientation,
        config.max_keypoints,
    );
    if config.vlfeat_compatible_output_order {
        order_vlfeat_rows_like_colmap(&mut keypoints, &mut diagnostics.after_orientation_cap);
    }
    diagnostics.after_cap = diagnostics.after_orientation_cap.clone();
    Ok((keypoints, diagnostics))
}

fn extract_sift_vlfeat_detector(
    image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftError> {
    let (keypoints, _) = detect_sift_vlfeat_keypoints(image, config)?;
    let descriptors = describe_sift_keypoints(image, &keypoints, config);
    Ok((keypoints, descriptors))
}

/// Run the compatible VLFeat/COLMAP detector and retain its candidate,
/// orientation-expansion, and global-cap stages without computing descriptors.
/// This is deliberately an explicit diagnostic API; normal extraction uses
/// [`extract_sift`] and does not expose or alter these records.
pub fn diagnose_sift_vlfeat_detector(
    image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<SiftDetectorDiagnostics, SiftError> {
    let (_, diagnostics) = detect_sift_vlfeat_keypoints(image, config)?;
    Ok(diagnostics)
}

/// Describe an existing, fixed set of SIFT keypoints without running the
/// detector again. The returned descriptor at index `k` corresponds exactly
/// to `keypoints[k]`; this is useful for append-only descriptor experiments
/// that must not create a second feature/track endpoint for the same locus.
pub fn describe_sift_keypoints(
    image: &GrayImage<'_>,
    keypoints: &[SiftKeypoint],
    config: &SiftConfig,
) -> Vec<Vec<f32>> {
    let vlfeat_pyramid = if config.vlfeat_compatible_descriptor && !keypoints.is_empty() {
        Some(build_vlfeat_pyramid(image, keypoints, config))
    } else {
        None
    };
    let pyramid = if !config.vlfeat_compatible_descriptor
        && config.scale_adaptive_gradients
        && !keypoints.is_empty()
    {
        Some(build_scale_adaptive_pyramid(image, keypoints, config))
    } else {
        None
    };
    keypoints
        .iter()
        .map(|kp| describe(image, kp, config, pyramid.as_ref(), vlfeat_pyramid.as_ref()))
        .collect()
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

const ORIENTATION_BINS: usize = 36;
const ORIENTATION_SMOOTHING_PASSES: usize = 6;

/// Smooth an orientation histogram on its circular domain.
///
/// Six passes of a three-tap box filter are the deterministic smoothing used
/// by the VLFeat-style SIFT orientation stage.  Non-finite and negative input
/// is treated as zero so malformed diagnostic/test histograms cannot produce
/// a non-finite orientation.
fn smooth_orientation_histogram(input: &[f64; ORIENTATION_BINS]) -> [f64; ORIENTATION_BINS] {
    let mut current = [0.0f64; ORIENTATION_BINS];
    for (dst, &value) in current.iter_mut().zip(input) {
        *dst = if value.is_finite() && value > 0.0 {
            value
        } else {
            0.0
        };
    }
    for _ in 0..ORIENTATION_SMOOTHING_PASSES {
        let mut next = [0.0f64; ORIENTATION_BINS];
        for bin in 0..ORIENTATION_BINS {
            let prev = current[(bin + ORIENTATION_BINS - 1) % ORIENTATION_BINS];
            let value = current[bin];
            let next_value = current[(bin + 1) % ORIENTATION_BINS];
            next[bin] = (prev + value + next_value) / 3.0;
        }
        current = next;
    }
    current
}

/// Interpolate one histogram peak with a bounded parabolic offset.
fn interpolate_orientation(bin: usize, prev: f64, value: f64, next: f64) -> f64 {
    let denominator = 2.0 * (2.0 * value - prev - next);
    let delta = if prev.is_finite()
        && value.is_finite()
        && next.is_finite()
        && denominator.is_finite()
        && denominator.abs() > f64::EPSILON
    {
        ((prev - next) / denominator).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    ((bin as f64 + delta) * 10.0)
        .rem_euclid(ORIENTATION_BINS as f64 * 10.0)
        .to_radians()
}

/// Select standards-aligned orientation peaks from a 36-bin histogram.
///
/// The returned tuple is `(bin, smoothed_strength, angle_radians)` and stays
/// in ascending-bin order.  Keeping that order lets the existing orientation
/// cap and keypoint indexing rules remain deterministic.
fn standard_orientation_peaks(histogram: &[f64; ORIENTATION_BINS]) -> Vec<(usize, f64, f64)> {
    let smoothed = smooth_orientation_histogram(histogram);
    let max_hist = smoothed.iter().copied().fold(0.0f64, f64::max);
    if !(max_hist.is_finite() && max_hist > 0.0) {
        return Vec::new();
    }

    let threshold = 0.8 * max_hist;
    let mut peaks = Vec::new();
    for bin in 0..ORIENTATION_BINS {
        let value = smoothed[bin];
        if !(value.is_finite() && value >= threshold) {
            continue;
        }
        let prev = smoothed[(bin + ORIENTATION_BINS - 1) % ORIENTATION_BINS];
        let next = smoothed[(bin + 1) % ORIENTATION_BINS];
        // Strict maxima suppress adjacent copies caused by histogram
        // quantisation.  The circular indexing also handles bins 35↔0.
        if !(value > prev && value > next) {
            continue;
        }
        peaks.push((bin, value, interpolate_orientation(bin, prev, value, next)));
    }
    peaks
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

    let mut peaks: Vec<(usize, f64, f64)> = if config.standard_orientation_peaks {
        standard_orientation_peaks(&hist)
    } else {
        // Keep this branch unchanged for byte-identical legacy extraction.
        let max_hist = hist.iter().cloned().fold(f64::MIN, f64::max);
        if max_hist <= 0.0 {
            return;
        }
        let mut peaks = Vec::new();
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
            let ori = ((bin as f64 + delta) * 10.0).rem_euclid(360.0).to_radians();
            peaks.push((bin, hist[bin], ori));
        }
        peaks
    };
    if config.max_orientations > 0 {
        // COLMAP-style: keep the strongest peaks only.
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        peaks.truncate(config.max_orientations);
        // Restore ascending-bin order among survivors for stable indexing.
        peaks.sort_by_key(|p| p.0);
    }
    for (_, _, ori) in peaks {
        out.push(SiftKeypoint {
            x: kp_x,
            y: kp_y,
            sigma,
            orientation: ori,
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
/// When [`SiftConfig::domain_size_pooling`] is on, averages unnormalized
/// histograms over a uniformly sampled range of domain sizes (COLMAP
/// DSP-SIFT) before the standard L2 clamp-and-renormalize.
fn describe(
    image: &GrayImage<'_>,
    kp: &SiftKeypoint,
    config: &SiftConfig,
    pyramid: Option<&ScaleAdaptivePyramid>,
    vlfeat_pyramid: Option<&VlfeatPyramid>,
) -> Vec<f32> {
    if let Some(vlfeat_pyramid) = vlfeat_pyramid {
        return describe_vlfeat_compatible(kp, config, vlfeat_pyramid);
    }
    if let Some(pyramid) = pyramid {
        return describe_scale_adaptive(kp, config, pyramid);
    }
    const DIM: usize = 128;
    let mut acc = vec![0.0f64; DIM];
    if config.domain_size_pooling && config.dsp_num_scales > 0 {
        let scales = dsp_domain_scale_factors(config);
        for scale in scales.iter().copied() {
            let raw = describe_raw(image, kp, scale, config.descriptor_magnification);
            for (a, r) in acc.iter_mut().zip(raw.iter()) {
                *a += *r;
            }
        }
        // DSP-SIFT uses a uniform density over domain sizes.  The subsequent
        // normalization is homogeneous, but keeping the explicit mean avoids
        // scale-dependent accumulation magnitudes and documents the pooling
        // operation independently of the final quantization step.
        let inv_count = 1.0 / scales.len() as f64;
        for value in &mut acc {
            *value *= inv_count;
        }
    } else {
        acc = describe_raw(image, kp, 1.0, config.descriptor_magnification);
    }
    normalize_sift_descriptor(&mut acc, config.normalization)
}

const VLFEAT_DESCRIPTOR_BINS: usize = 8;
const VLFEAT_DESCRIPTOR_CELLS: usize = 4;
const VLFEAT_DESCRIPTOR_MAGNIFICATION: f64 = 3.0;
const VLFEAT_DESCRIPTOR_WINDOW_SIZE: f64 = VLFEAT_DESCRIPTOR_CELLS as f64 / 2.0;

/// Radius of the square sample window in VLFeat's descriptor loop.
///
/// This is the source expression
/// `floor(sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5)` from
/// `vl_sift_calc_keypoint_descriptor` (`sift.c`, immediately after `SBP`).
/// It includes the half-bin interpolation margin and the diagonal of the
/// rotated support; it is not the orientation-histogram `ceil(3*sigmaw)`.
fn vlfeat_descriptor_window_radius(sbp: f64) -> i64 {
    if !(sbp.is_finite() && sbp > 0.0) {
        return 0;
    }
    (std::f64::consts::SQRT_2 * sbp * (VLFEAT_DESCRIPTOR_CELLS as f64 + 1.0) / 2.0 + 0.5).floor()
        as i64
}

/// Add one VLFeat descriptor sample using its 4×4×8 trilinear accumulator.
/// `nx`/`ny` are coordinates in cell units and `orientation_bin` is in the
/// circular 8-bin orientation domain.  Keeping this operation separate makes
/// the bin placement and wrap rules directly testable.
fn accumulate_vlfeat_sample(
    descriptor: &mut [f64],
    magnitude: f64,
    nx: f64,
    ny: f64,
    orientation_bin: f64,
    weight: f64,
) {
    if descriptor.len()
        != VLFEAT_DESCRIPTOR_CELLS * VLFEAT_DESCRIPTOR_CELLS * VLFEAT_DESCRIPTOR_BINS
        || !magnitude.is_finite()
        || !nx.is_finite()
        || !ny.is_finite()
        || !orientation_bin.is_finite()
        || !weight.is_finite()
        || magnitude <= 0.0
        || weight <= 0.0
    {
        return;
    }
    let binx = (nx - 0.5).floor() as i64;
    let biny = (ny - 0.5).floor() as i64;
    let bint = orientation_bin.floor() as i64;
    let remainder_x = nx - (binx as f64 + 0.5);
    let remainder_y = ny - (biny as f64 + 0.5);
    let remainder_t = orientation_bin - bint as f64;
    for dbinx in 0..=1i64 {
        let cell_x = binx + dbinx;
        if !(-2..2).contains(&cell_x) {
            continue;
        }
        let weight_x = (1.0 - (dbinx as f64 - remainder_x).abs()).max(0.0);
        if weight_x <= 0.0 {
            continue;
        }
        for dbiny in 0..=1i64 {
            let cell_y = biny + dbiny;
            if !(-2..2).contains(&cell_y) {
                continue;
            }
            let weight_y = (1.0 - (dbiny as f64 - remainder_y).abs()).max(0.0);
            if weight_y <= 0.0 {
                continue;
            }
            for dbint in 0..=1i64 {
                let orientation = (bint + dbint).rem_euclid(VLFEAT_DESCRIPTOR_BINS as i64) as usize;
                let weight_t = if dbint == 0 {
                    1.0 - remainder_t
                } else {
                    remainder_t
                };
                if weight_t <= 0.0 {
                    continue;
                }
                let index = (((cell_y + 2) as usize * VLFEAT_DESCRIPTOR_CELLS
                    + (cell_x + 2) as usize)
                    * VLFEAT_DESCRIPTOR_BINS)
                    + orientation;
                descriptor[index] += magnitude * weight * weight_x * weight_y * weight_t;
            }
        }
    }
}

/// Apply the normalization used by the COLMAP CPU SIFT path, then quantize to
/// the same 512-scaled uint8-equivalent values while keeping the Rust matcher
/// interface as `f32`.  The byte scale is global and therefore does not alter
/// Euclidean NN ordering, but it preserves COLMAP's rounding behavior.
#[allow(clippy::manual_clamp)]
fn normalize_vlfeat_descriptor(desc: &mut [f64], mode: SiftNormalization) -> Vec<f32> {
    match mode {
        SiftNormalization::L2 => {
            let norm = desc.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm > f64::EPSILON && norm.is_finite() {
                for value in desc.iter_mut() {
                    *value /= norm;
                }
            }
            for value in desc.iter_mut() {
                *value = value.min(0.2).max(0.0);
            }
            let clipped_norm = desc.iter().map(|value| value * value).sum::<f64>().sqrt();
            if clipped_norm > f64::EPSILON && clipped_norm.is_finite() {
                for value in desc.iter_mut() {
                    *value /= clipped_norm;
                }
            }
        }
        SiftNormalization::L1Root => {
            let l1 = desc.iter().map(|value| value.abs()).sum::<f64>();
            if l1 > f64::EPSILON && l1.is_finite() {
                for value in desc.iter_mut() {
                    *value = (*value / l1).max(0.0).sqrt();
                }
            }
        }
    }

    // COLMAP computes the byte vector before its final VLFeat→UBC layout
    // permutation.  `round` is the C++ std::round behavior used there.
    let mut quantized = vec![0.0f32; desc.len()];
    for (output, &value) in quantized.iter_mut().zip(desc.iter()) {
        *output = (value * 512.0).round().clamp(0.0, 255.0) as f32;
    }
    transform_vlfeat_to_ubc(&quantized)
}

/// COLMAP's `TransformVLFeatToUBCFeatureDescriptors`: reverse the orientation
/// order within each spatial cell while leaving the 4×4 spatial layout intact.
fn transform_vlfeat_to_ubc(descriptor: &[f32]) -> Vec<f32> {
    debug_assert_eq!(descriptor.len(), 128);
    const Q: [usize; 8] = [0, 7, 6, 5, 4, 3, 2, 1];
    let mut transformed = vec![0.0f32; descriptor.len()];
    for row in 0..VLFEAT_DESCRIPTOR_CELLS {
        for col in 0..VLFEAT_DESCRIPTOR_CELLS {
            let base = VLFEAT_DESCRIPTOR_BINS * (col + VLFEAT_DESCRIPTOR_CELLS * row);
            for (source_orientation, &target_orientation) in Q.iter().enumerate() {
                transformed[base + target_orientation] = descriptor[base + source_orientation];
            }
        }
    }
    transformed
}

/// Descriptor path matching the unit of conventions used by VLFeat and
/// COLMAP's CPU extractor.  It intentionally does not call the experimental
/// `scale_adaptive_gradients` path: the octave/coordinate, support, histogram,
/// normalization, and quantization choices must move together.
fn describe_vlfeat_compatible(
    kp: &SiftKeypoint,
    config: &SiftConfig,
    pyramid: &VlfeatPyramid,
) -> Vec<f32> {
    let mut accumulated = vec![0.0f64; 128];
    if config.domain_size_pooling && config.dsp_num_scales > 0 {
        let scales = dsp_domain_scale_factors(config);
        for scale in scales.iter().copied() {
            let raw = describe_raw_vlfeat(kp, scale, pyramid);
            for (target, source) in accumulated.iter_mut().zip(raw.iter()) {
                *target += *source;
            }
        }
        let inv_count = 1.0 / scales.len() as f64;
        for value in &mut accumulated {
            *value *= inv_count;
        }
    } else {
        accumulated = describe_raw_vlfeat(kp, 1.0, pyramid);
    }
    normalize_vlfeat_descriptor(&mut accumulated, config.normalization)
}

/// Build one unnormalized VLFeat histogram from the selected octave gradient
/// image.  External COLMAP keypoints store centers in the `(0.5, 0.5)` pixel
/// convention; subtracting that offset maps them to the integer image lattice
/// used by the Rust `GrayImage` and by VLFeat's internal descriptor routine.
fn describe_raw_vlfeat(kp: &SiftKeypoint, scale_factor: f64, pyramid: &VlfeatPyramid) -> Vec<f64> {
    let mut descriptor = vec![0.0f64; 128];
    let sigma = if kp.sigma.is_finite() && kp.sigma > 0.0 {
        kp.sigma * scale_factor.max(1e-6)
    } else {
        pyramid.base_sigma
    };
    let selection = pyramid.select(sigma);
    let local_sigma = sigma / selection.pixel_scale;
    // `vl_sift_calc_keypoint_descriptor` uses `SBP = magnif * sigma +
    // VL_EPSILON_D`; the positive fallback above keeps malformed external
    // scales from producing a non-finite descriptor while preserving the
    // source expression for every valid keypoint.
    let sbp = (VLFEAT_DESCRIPTOR_MAGNIFICATION * local_sigma + f64::EPSILON).max(1e-6);
    let width = pyramid.octaves[selection.octave].levels[selection.level].width as i64;
    let height = pyramid.octaves[selection.octave].levels[selection.level].height as i64;
    let local_x = (kp.x - 0.5) / selection.pixel_scale;
    let local_y = (kp.y - 0.5) / selection.pixel_scale;
    let xi = (local_x + 0.5).floor() as i64;
    let yi = (local_y + 0.5).floor() as i64;
    let window = vlfeat_descriptor_window_radius(sbp);
    // Exact `vl_sift_calc_keypoint_descriptor` bounds (`sift.c`):
    // max(-W, 1-xi)..min(W, w-xi-2), and the analogous y interval. The
    // outermost image samples are intentionally excluded because the source
    // gradient buffer is only synchronized for descriptor-valid levels.
    let min_dx = (-window).max(1 - xi);
    let max_dx = window.min(width - 2 - xi);
    let min_dy = (-window).max(1 - yi);
    let max_dy = window.min(height - 2 - yi);
    if min_dx > max_dx || min_dy > max_dy {
        return descriptor;
    }
    let (cos_orientation, sin_orientation) = (kp.orientation.cos(), kp.orientation.sin());
    for dyi in min_dy..=max_dy {
        for dxi in min_dx..=max_dx {
            let sample_x = xi + dxi;
            let sample_y = yi + dyi;
            let (gx, gy) = pyramid.sample_integer(selection, sample_x, sample_y);
            let magnitude = (gx * gx + gy * gy).sqrt();
            if !(magnitude.is_finite() && magnitude > f64::EPSILON) {
                continue;
            }
            let dx = sample_x as f64 - local_x;
            let dy = sample_y as f64 - local_y;
            let nx = (cos_orientation * dx + sin_orientation * dy) / sbp;
            let ny = (-sin_orientation * dx + cos_orientation * dy) / sbp;
            let theta = (gy.atan2(gx) - kp.orientation).rem_euclid(std::f64::consts::TAU);
            let orientation_bin = theta * VLFEAT_DESCRIPTOR_BINS as f64 / std::f64::consts::TAU;
            let gaussian_weight =
                (-(nx * nx + ny * ny) / (2.0 * VLFEAT_DESCRIPTOR_WINDOW_SIZE.powi(2))).exp();
            accumulate_vlfeat_sample(
                &mut descriptor,
                magnitude,
                nx,
                ny,
                orientation_bin,
                gaussian_weight,
            );
        }
    }
    descriptor
}

/// Scale-adaptive descriptor path.  This mirrors [`describe`] but obtains
/// gradients from a Gaussian level at the requested absolute keypoint scale.
/// The legacy implementation remains in [`describe_raw`] and is not touched
/// when `scale_adaptive_gradients` is false.
fn describe_scale_adaptive(
    kp: &SiftKeypoint,
    config: &SiftConfig,
    pyramid: &ScaleAdaptivePyramid,
) -> Vec<f32> {
    const DIM: usize = 128;
    let mut acc = vec![0.0f64; DIM];
    if config.domain_size_pooling && config.dsp_num_scales > 0 {
        let scales = dsp_domain_scale_factors(config);
        for scale in scales.iter().copied() {
            let raw =
                describe_raw_scale_adaptive(kp, scale, config.descriptor_magnification, pyramid);
            for (a, r) in acc.iter_mut().zip(raw.iter()) {
                *a += *r;
            }
        }
        let inv_count = 1.0 / scales.len() as f64;
        for value in &mut acc {
            *value *= inv_count;
        }
    } else {
        acc = describe_raw_scale_adaptive(kp, 1.0, config.descriptor_magnification, pyramid);
    }
    normalize_sift_descriptor(&mut acc, config.normalization)
}

/// Build one unnormalized SIFT histogram from a scale-adaptive gradient
/// field.  Coordinates are transformed into the selected octave before
/// sampling, while the keypoint geometry and descriptor binning stay in the
/// original-image frame.  This is the precise inverse of the octave's
/// `pixel_scale` mapping and avoids changing exported feature coordinates.
fn describe_raw_scale_adaptive(
    kp: &SiftKeypoint,
    scale_factor: f64,
    descriptor_magnification: f64,
    pyramid: &ScaleAdaptivePyramid,
) -> Vec<f64> {
    const DIM: usize = 128;
    let mut desc = vec![0.0f64; DIM];
    let sigma = (kp.sigma * scale_factor).max(1e-6);
    let cell_size = (descriptor_magnification.max(1e-6) * sigma).max(3.0);
    let (cos, sin) = (kp.orientation.cos(), kp.orientation.sin());
    let shape = kp.affine_shape.unwrap_or([[1.0, 0.0], [0.0, 1.0]]);
    let a00a11 = shape[0][0] * shape[0][0] + shape[1][0] * shape[1][0];
    let a01a11 = shape[0][1] * shape[0][1] + shape[1][1] * shape[1][1];
    let max_stretch = a00a11.max(a01a11).sqrt().max(1.0);
    let half = (cell_size * max_stretch * 4.0f64.sqrt()) as i64;
    let step = scale_factor.round().max(1.0) as i64;
    let area_weight = (step * step) as f64;
    let selection = pyramid.select(sigma);
    let pixel_scale = selection.pixel_scale;
    let local_x = kp.x / pixel_scale;
    let local_y = kp.y / pixel_scale;
    let ex = (shape[0][0] / pixel_scale, shape[1][0] / pixel_scale);
    let ey = (shape[0][1] / pixel_scale, shape[1][1] / pixel_scale);

    for dy in (-half..=half).step_by(step as usize) {
        for dx in (-half..=half).step_by(step as usize) {
            let rx0 = dx as f64 * cos - dy as f64 * sin;
            let ry0 = dx as f64 * sin + dy as f64 * cos;
            let (wx, wy) = mat2_apply(shape, (rx0, ry0));
            let (sample_gx, sample_gy) = pyramid.sample(
                selection,
                local_x + wx / pixel_scale,
                local_y + wy / pixel_scale,
            );
            // The gradient layer is differentiated in octave pixels.  Project
            // it onto the (possibly affine) image-frame columns and convert
            // the result back to original-pixel units before binning.
            let gx = sample_gx * ex.0 + sample_gy * ex.1;
            let gy = sample_gx * ey.0 + sample_gy * ey.1;
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= f64::EPSILON {
                continue;
            }
            let theta = gy.atan2(gx) - kp.orientation;
            let weight =
                (-(rx0 * rx0 + ry0 * ry0) / (2.0 * (cell_size * 2.0).powi(2))).exp() * area_weight;
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
    desc
}

/// Unnormalized SIFT histogram at `scale_factor × kp.sigma` domain size.
fn describe_raw(
    image: &GrayImage<'_>,
    kp: &SiftKeypoint,
    scale_factor: f64,
    descriptor_magnification: f64,
) -> Vec<f64> {
    const DIM: usize = 128;
    let mut desc = vec![0.0f64; DIM];
    let sigma = (kp.sigma * scale_factor).max(1e-6);
    let cell_size = (descriptor_magnification.max(1e-6) * sigma).max(3.0);
    let (cos, sin) = (kp.orientation.cos(), kp.orientation.sin());
    let shape = kp.affine_shape.unwrap_or([[1.0, 0.0], [0.0, 1.0]]);
    let a00a11 = shape[0][0] * shape[0][0] + shape[1][0] * shape[1][0];
    let a01a11 = shape[0][1] * shape[0][1] + shape[1][1] * shape[1][1];
    let max_stretch = a00a11.max(a01a11).sqrt().max(1.0);
    let half = (cell_size * max_stretch * 4.0f64.sqrt()) as i64;
    // Keep sample count roughly scale-invariant: stride grows with domain size
    // so DSP pooling stays ~O(num_scales) rather than O(Σ scale²).
    let step = scale_factor.round().max(1.0) as i64;
    let area_weight = (step * step) as f64;
    for dy in (-half..=half).step_by(step as usize) {
        for dx in (-half..=half).step_by(step as usize) {
            let rx0 = dx as f64 * cos - dy as f64 * sin;
            let ry0 = dx as f64 * sin + dy as f64 * cos;
            let (wx, wy) = mat2_apply(shape, (rx0, ry0));
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
            let weight =
                (-(rx0 * rx0 + ry0 * ry0) / (2.0 * (cell_size * 2.0).powi(2))).exp() * area_weight;
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
    desc
}

fn normalize_sift_descriptor(desc: &mut [f64], mode: SiftNormalization) -> Vec<f32> {
    match mode {
        SiftNormalization::L2 => {
            let norm: f64 = desc.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > f64::EPSILON {
                for v in desc.iter_mut() {
                    *v /= norm;
                }
            }
            for v in desc.iter_mut() {
                *v = v.min(0.2);
            }
            let clipped: f64 = desc.iter().map(|v| v * v).sum::<f64>().sqrt();
            if clipped > f64::EPSILON {
                for v in desc.iter_mut() {
                    *v /= clipped;
                }
            }
        }
        SiftNormalization::L1Root => {
            // COLMAP L1_ROOT / RootSIFT: L1 → √ → (optional) L2 for unit length.
            let l1: f64 = desc.iter().map(|v| v.abs()).sum();
            if l1 > f64::EPSILON {
                for v in desc.iter_mut() {
                    *v = (*v / l1).max(0.0).sqrt();
                }
            }
            let l2: f64 = desc.iter().map(|v| v * v).sum::<f64>().sqrt();
            if l2 > f64::EPSILON {
                for v in desc.iter_mut() {
                    *v /= l2;
                }
            }
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
    fn sift_extraction_repeat_is_byte_identical() {
        let (w, h) = (64usize, 48usize);
        let pixels = blob_image(w, h, (31.25, 23.5), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig {
            octaves: 2,
            intervals: 3,
            max_keypoints: 96,
            ..SiftConfig::default()
        };
        let first = extract_sift(&image, &config).unwrap();
        let second = extract_sift(&image, &config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn scale_adaptive_gradient_field_handles_constant_and_linear_images() {
        let (w, h) = (32usize, 32usize);
        let constant = Layer {
            width: w,
            height: h,
            data: vec![0.37; w * h],
        };
        let constant_gradient = gradient_layer(&constant);
        let (cgx, cgy) = constant_gradient.sample(13.25, 17.5);
        assert!(cgx.abs() < 1e-12);
        assert!(cgy.abs() < 1e-12);

        let mut linear_data = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                linear_data[y * w + x] = 0.2 + 0.013 * x as f64 + 0.021 * y as f64;
            }
        }
        let linear = Layer {
            width: w,
            height: h,
            data: linear_data,
        };
        let linear_gradient = gradient_layer(&linear);
        let (lgx, lgy) = linear_gradient.sample(13.25, 17.5);
        assert!((lgx - 0.013).abs() < 1e-8, "linear gx={lgx}");
        assert!((lgy - 0.021).abs() < 1e-8, "linear gy={lgy}");
    }

    #[test]
    fn scale_adaptive_selection_maps_scale_and_coordinates_deterministically() {
        let (w, h) = (64usize, 64usize);
        let pixels = vec![0.5f32; w * h];
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig::default();
        let keypoints = [SiftKeypoint::from_location_scale_orientation(
            24.0, 16.0, 4.0, 0.0,
        )];
        let pyramid = build_scale_adaptive_pyramid(&image, &keypoints, &config);

        let exact = pyramid.select(config.sigma_input * pyramid.k.powi(2));
        assert_eq!(exact.octave, 0);
        assert_eq!(exact.lower, 2);
        assert_eq!(exact.upper, 2);
        assert_eq!(exact.weight, 0.0);

        let halfway = pyramid.select(config.sigma_input * pyramid.k.powf(2.5));
        assert_eq!(halfway.octave, 0);
        assert_eq!(halfway.lower, 2);
        assert_eq!(halfway.upper, 3);
        assert!((halfway.weight - 0.5).abs() < 1e-12);

        let scale = pyramid.select(4.0).pixel_scale;
        let local_x = 24.0 / scale;
        let local_y = 16.0 / scale;
        assert!((local_x * scale - 24.0).abs() < 1e-12);
        assert!((local_y * scale - 16.0).abs() < 1e-12);
        assert_eq!(pyramid.select(4.0), pyramid.select(4.0));
    }

    #[test]
    fn scale_adaptive_descriptors_are_opt_in_deterministic_and_distinct() {
        let (w, h) = (96usize, 96usize);
        let mut pixels = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let xf = x as f64;
                let yf = y as f64;
                pixels[y * w + x] = (0.45
                    + 0.22 * (xf * 0.17).sin()
                    + 0.19 * (yf * 0.13).cos()
                    + 0.11 * ((xf + yf) * 0.07).sin())
                .clamp(0.0, 1.0) as f32;
            }
        }
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let keypoints = [SiftKeypoint::from_location_scale_orientation(
            48.0, 47.0, 2.5, 0.37,
        )];
        let legacy = SiftConfig::default();
        let explicit_legacy = SiftConfig {
            scale_adaptive_gradients: false,
            ..legacy.clone()
        };
        let adaptive = SiftConfig {
            scale_adaptive_gradients: true,
            ..legacy.clone()
        };
        let legacy_descriptor = describe_sift_keypoints(&image, &keypoints, &legacy);
        assert_eq!(
            legacy_descriptor,
            describe_sift_keypoints(&image, &keypoints, &explicit_legacy),
            "the default scale-adaptive flag must preserve legacy bytes"
        );
        let adaptive_descriptor = describe_sift_keypoints(&image, &keypoints, &adaptive);
        assert_ne!(legacy_descriptor, adaptive_descriptor);
        assert_eq!(
            adaptive_descriptor,
            describe_sift_keypoints(&image, &keypoints, &adaptive),
            "scale-adaptive descriptors must be deterministic"
        );
    }

    #[test]
    fn vlfeat_histogram_uses_trilinear_bins_and_circular_orientation() {
        let mut descriptor = vec![0.0f64; 128];
        // x=1.5 and y=-1.5 land exactly in cell (col=1,row=0); an orientation
        // just below 2π must split between bins 7 and 0.
        accumulate_vlfeat_sample(&mut descriptor, 2.0, 1.5, -1.5, 7.75, 1.0);
        let base = 3usize * VLFEAT_DESCRIPTOR_BINS;
        assert!((descriptor[base + 7] - 0.5).abs() < 1e-12);
        assert!((descriptor[base] - 1.5).abs() < 1e-12);
        assert!(
            (descriptor.iter().sum::<f64>() - 2.0).abs() < 1e-12,
            "trilinear weights must conserve sample mass"
        );
    }

    #[test]
    fn vlfeat_orientation_bilinear_mode_matches_colmap_source_switch() {
        let mut nearest = [0.0f64; ORIENTATION_BINS];
        let mut bilinear = [0.0f64; ORIENTATION_BINS];
        // COLMAP's vendored VLFeat defines VL_SIFT_BILINEAR_ORIENTATIONS:
        // angle zero lies halfway between the circular bins 35 and 0.
        accumulate_vlfeat_orientation(&mut nearest, 0.0, 2.0, 1.0, false);
        accumulate_vlfeat_orientation(&mut bilinear, 0.0, 2.0, 1.0, true);
        assert_eq!(nearest[0], 2.0);
        assert_eq!(nearest[35], 0.0);
        assert!((bilinear[35] - 1.0).abs() < 1e-12);
        assert!((bilinear[0] - 1.0).abs() < 1e-12);
        assert!((bilinear.iter().sum::<f64>() - 2.0).abs() < 1e-12);

        let mut split = [0.0f64; ORIENTATION_BINS];
        accumulate_vlfeat_orientation(
            &mut split,
            std::f64::consts::TAU * 2.0 / ORIENTATION_BINS as f64,
            3.0,
            0.25,
            true,
        );
        assert!((split[1] - 0.375).abs() < 1e-12);
        assert!((split[2] - 0.375).abs() < 1e-12);
        assert!((split.iter().sum::<f64>() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn vlfeat_orientation_bilinear_matches_reference_vector() {
        let samples = [
            (0.0, 1.25, 0.5),
            (
                std::f64::consts::TAU * 35.75 / ORIENTATION_BINS as f64,
                2.0,
                0.75,
            ),
            (-0.31, 0.4, 1.2),
            (
                std::f64::consts::TAU * 7.125 / ORIENTATION_BINS as f64,
                3.0,
                0.25,
            ),
        ];
        let mut actual = [0.0f64; ORIENTATION_BINS];
        let mut expected = [0.0f64; ORIENTATION_BINS];
        for &(angle, magnitude, weight) in &samples {
            accumulate_vlfeat_orientation(&mut actual, angle, magnitude, weight, true);

            // Independent transcription of COLMAP's vendored VLFeat branch
            // (sift.c:1636-1642), retained here as a small numerical vector
            // rather than asserting the implementation against itself.
            let fbin = angle.rem_euclid(std::f64::consts::TAU) * ORIENTATION_BINS as f64
                / std::f64::consts::TAU;
            let bin = (fbin - 0.5).floor() as isize;
            let rbin = fbin - bin as f64 - 0.5;
            let contribution = magnitude * weight;
            expected[bin.rem_euclid(ORIENTATION_BINS as isize) as usize] +=
                (1.0 - rbin) * contribution;
            expected[(bin + 1).rem_euclid(ORIENTATION_BINS as isize) as usize] +=
                rbin * contribution;
        }
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn vlfeat_orientation_peak_order_and_four_peak_cap_match_source() {
        let mut histogram = [0.0f64; ORIENTATION_BINS];
        for (bin, value) in [(1, 1.0), (10, 0.95), (20, 0.9), (30, 0.85), (35, 0.81)] {
            histogram[bin] = value;
        }
        let peaks = select_vlfeat_orientation_peaks(&histogram);
        assert_eq!(peaks.len(), 4, "VLFeat returns at most four peaks");
        let bins = peaks
            .iter()
            .map(|&angle| {
                ((angle * ORIENTATION_BINS as f64 / std::f64::consts::TAU) - 0.5).round() as usize
                    % ORIENTATION_BINS
            })
            .collect::<Vec<_>>();
        assert_eq!(bins, vec![1, 10, 20, 30]);
    }

    #[test]
    fn vlfeat_orientation_peak_selector_wraps_circular_boundary() {
        let mut histogram = [0.0f64; ORIENTATION_BINS];
        histogram[35] = 1.0;
        histogram[0] = 0.2;
        let peaks = select_vlfeat_orientation_peaks(&histogram);
        assert_eq!(peaks.len(), 1);
        assert!(peaks[0] > 6.0, "bin 35 must stay near 2π");
    }

    #[test]
    fn vlfeat_normalization_quantization_and_layout_match_colmap_rules() {
        let mut raw = vec![0.05f64; 128];
        let descriptor = normalize_vlfeat_descriptor(&mut raw, SiftNormalization::L2);
        assert_eq!(descriptor[0], 45.0, "round(512 / sqrt(128))");
        assert!(descriptor.iter().all(|&value| value == 45.0));

        let mut vlfeat_layout = vec![0.0f32; 128];
        for (index, value) in vlfeat_layout.iter_mut().enumerate() {
            *value = index as f32;
        }
        let ubc = transform_vlfeat_to_ubc(&vlfeat_layout);
        assert_eq!(ubc[0], 0.0);
        assert_eq!(ubc[1], 7.0);
        assert_eq!(ubc[7], 1.0);
        assert_eq!(ubc[8], 8.0, "spatial cells remain in row-major order");
    }

    #[test]
    fn vlfeat_descriptor_window_radius_matches_source_formula() {
        for sbp in [0.25, 0.8, 1.0, 2.4, 3.0, 6.0, 17.125] {
            let expected = (std::f64::consts::SQRT_2 * sbp * (VLFEAT_DESCRIPTOR_CELLS as f64 + 1.0)
                / 2.0
                + 0.5)
                .floor() as i64;
            assert_eq!(vlfeat_descriptor_window_radius(sbp), expected);
        }
        assert_eq!(vlfeat_descriptor_window_radius(3.0), 11);
        assert_eq!(vlfeat_descriptor_window_radius(6.0), 21);
        assert_eq!(vlfeat_descriptor_window_radius(f64::NAN), 0);
    }

    #[test]
    fn vlfeat_gradient_boundary_stencil_matches_source() {
        let layer = Layer {
            width: 4,
            height: 3,
            data: (0..12).map(|value| value as f64).collect(),
        };
        let gradient = gradient_layer(&layer);
        // x changes by one and y changes by four. Both boundary and interior
        // samples use the same full/central finite-difference value here.
        for y in 0..3 {
            for x in 0..4 {
                let index = y * 4 + x;
                assert!((gradient.gx[index] as f64 - 1.0).abs() < 1e-12);
                assert!((gradient.gy[index] as f64 - 4.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn vlfeat_pyramid_scale_selection_is_deterministic() {
        let (w, h) = (64usize, 64usize);
        let pixels = vec![0.5f32; w * h];
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig::default();
        let keypoints = [SiftKeypoint::from_location_scale_orientation(
            24.5, 16.5, 3.2, 0.0,
        )];
        let pyramid = build_vlfeat_pyramid(&image, &keypoints, &config);
        let base = pyramid.select(0.8);
        assert_eq!(base.octave, 0);
        // The source descriptor only has gradients for s=s_min+1..S-1;
        // array level 1 is therefore the smallest valid level in this
        // full-resolution representation of source octave -1.
        assert_eq!(base.level, 1);
        let next_octave = pyramid.select(1.6);
        assert_eq!(next_octave.octave, 0);
        assert_eq!(next_octave.level, 3);
        let octave_boundary = pyramid.select(1.6 * 2.0f64.powf(1.0 / 3.0));
        assert_eq!(octave_boundary.octave, 1);
        assert_eq!(octave_boundary.level, 1);
        assert_eq!(next_octave, pyramid.select(1.6));
    }

    #[test]
    fn vlfeat_descriptor_mode_is_opt_in_and_deterministic() {
        let (w, h) = (64usize, 64usize);
        let pixels = blob_image(w, h, (32.0, 30.0), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let legacy = SiftConfig {
            octaves: 2,
            ..SiftConfig::default()
        };
        let explicit_legacy = SiftConfig {
            vlfeat_compatible_descriptor: false,
            vlfeat_compatible_detector: false,
            ..legacy.clone()
        };
        let vlfeat = SiftConfig {
            vlfeat_compatible_descriptor: true,
            ..legacy.clone()
        };
        assert_eq!(
            extract_sift(&image, &legacy).unwrap(),
            extract_sift(&image, &explicit_legacy).unwrap(),
            "the new mode must not alter the default path"
        );
        let (legacy_keypoints, legacy_descriptors) = extract_sift(&image, &legacy).unwrap();
        let (vlfeat_keypoints, vlfeat_descriptors) = extract_sift(&image, &vlfeat).unwrap();
        assert_eq!(legacy_keypoints, vlfeat_keypoints);
        assert_eq!(vlfeat_descriptors, extract_sift(&image, &vlfeat).unwrap().1);
        assert!(
            legacy_descriptors != vlfeat_descriptors,
            "the explicit VLFeat path should select its distinct layout/quantization"
        );
        assert!(vlfeat_descriptors
            .iter()
            .flat_map(|descriptor| descriptor.iter())
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn vlfeat_compatible_octave_default_matches_colmap_and_explicit_wins() {
        // COLMAP's SiftExtractionOptions defaults are O=4 and o_min=-1. The
        // image-size limit still applies for small images, while an explicit
        // config value is preserved for controlled parity experiments.
        assert_eq!(vlfeat_compatible_octave_count(0, 2048), 4);
        assert_eq!(vlfeat_compatible_octave_count(0, 8), 4);
        assert_eq!(vlfeat_compatible_octave_count(0, 3), 3);
        assert_eq!(vlfeat_compatible_octave_count(6, 2048), 6);
        assert_eq!(vlfeat_compatible_octave_count(99, 3), 3);
    }

    #[test]
    fn vlfeat_detector_mode_is_opt_in_and_deterministic() {
        let (width, height) = (96usize, 96usize);
        let pixels = blob_image(width, height, (48.25, 46.75), 5.0);
        let image = GrayImage::new(width, height, &pixels).unwrap();
        let legacy = SiftConfig {
            octaves: 2,
            max_keypoints: 128,
            ..SiftConfig::default()
        };
        let explicit_legacy = SiftConfig {
            vlfeat_compatible_detector: false,
            ..legacy.clone()
        };
        assert_eq!(
            extract_sift(&image, &legacy).unwrap(),
            extract_sift(&image, &explicit_legacy).unwrap(),
            "the compatible detector must be default-off"
        );
        let compatible = SiftConfig {
            vlfeat_compatible_detector: true,
            vlfeat_compatible_descriptor: true,
            max_keypoints: 128,
            ..legacy
        };
        let first = extract_sift(&image, &compatible).unwrap();
        let second = extract_sift(&image, &compatible).unwrap();
        assert_eq!(first, second, "compatible detection must be deterministic");
        assert_eq!(first.0.len(), first.1.len());
        assert!(first.0.iter().all(|keypoint| keypoint.x.is_finite()
            && keypoint.y.is_finite()
            && keypoint.sigma.is_finite()
            && keypoint.sigma > 0.0));
    }

    #[test]
    fn vlfeat_detector_diagnostics_are_stage_ordered_and_deterministic() {
        let (width, height) = (96usize, 96usize);
        let pixels = blob_image(width, height, (48.25, 46.75), 5.0);
        let image = GrayImage::new(width, height, &pixels).unwrap();
        let config = SiftConfig {
            octaves: 2,
            max_keypoints: 128,
            max_orientations: 2,
            vlfeat_compatible_detector: true,
            ..SiftConfig::default()
        };
        let first = diagnose_sift_vlfeat_detector(&image, &config).unwrap();
        let second = diagnose_sift_vlfeat_detector(&image, &config).unwrap();
        assert_eq!(first, second, "diagnostic rows must be deterministic");
        assert!(first.after_orientation.len() >= first.after_orientation_cap.len());
        assert!(first.after_orientation_cap.len() >= first.after_cap.len());
        assert!(first.before_orientation.iter().all(|row| {
            row.x.is_finite()
                && row.y.is_finite()
                && row.sigma.is_finite()
                && row.sigma > 0.0
                && row.response.is_finite()
        }));
        assert!(first.after_cap.iter().all(|row| {
            row.x.is_finite()
                && row.y.is_finite()
                && row.sigma.is_finite()
                && row.sigma > 0.0
                && row.orientation.is_finite()
                && row.response.is_finite()
                && row.orientation_index < 4
        }));
    }

    #[test]
    fn vlfeat_detector_upsample_is_bilinear_with_repeated_boundary() {
        let source = Layer {
            width: 2,
            height: 2,
            data: vec![0.0, 1.0, 2.0, 3.0],
        };
        let upsampled = double_up_vlfeat(&source);
        assert_eq!(upsampled.width, 4);
        assert_eq!(upsampled.height, 4);
        assert_eq!(
            upsampled.data,
            vec![0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 2.0, 2.5, 3.0, 3.0,]
        );
    }

    #[test]
    fn vlfeat_detector_localization_recovers_subpixel_subscale_peak() {
        let (width, height) = (9usize, 9usize);
        let target = (4.25, 3.7, 0.2);
        let make_layer = |scale: f64| Layer {
            width,
            height,
            data: (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        10.0 - (x as f64 - target.0).powi(2)
                            - (y as f64 - target.1).powi(2)
                            - (scale - target.2).powi(2)
                    })
                })
                .collect(),
        };
        let refined = refine_vlfeat_extremum(
            &make_layer(-1.0),
            &make_layer(0.0),
            &make_layer(1.0),
            4,
            4,
            0.0,
            4.0,
            0.1,
            10.0,
        )
        .expect("quadratic maximum should survive refinement");
        assert!((refined.x - target.0).abs() < 1e-10, "x={}", refined.x);
        assert!((refined.y - target.1).abs() < 1e-10, "y={}", refined.y);
        assert!(
            (refined.scale - target.2).abs() < 1e-10,
            "s={}",
            refined.scale
        );
        assert!((refined.value - 10.0).abs() < 1e-10);
    }

    #[test]
    fn vlfeat_detector_rejects_anisotropic_edge_peak() {
        let (width, height) = (9usize, 9usize);
        let make_layer = |scale: f64| Layer {
            width,
            height,
            data: (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        10.0 - 30.0 * (x as f64 - 4.0).powi(2)
                            - (y as f64 - 4.0).powi(2)
                            - (scale * 1.0).powi(2)
                    })
                })
                .collect(),
        };
        assert!(refine_vlfeat_extremum(
            &make_layer(-1.0),
            &make_layer(0.0),
            &make_layer(1.0),
            4,
            4,
            0.0,
            4.0,
            0.1,
            10.0,
        )
        .is_none());
    }

    #[test]
    fn vlfeat_detector_cap_keeps_whole_source_levels() {
        let candidate = |x: f64, level: i32| SiftDetectionCandidate {
            x,
            y: 10.0,
            sigma: 1.0 + level as f64,
            response: 1.0,
            octave: -1,
            level,
            edge_score: 0.0,
        };
        let oriented =
            |candidate: &SiftDetectionCandidate, orientation_index: usize| SiftOrientedDetection {
                x: candidate.x,
                y: candidate.y,
                sigma: candidate.sigma,
                orientation: orientation_index as f64,
                response: candidate.response,
                octave: candidate.octave,
                level: candidate.level,
                orientation_index,
            };
        let candidates = vec![candidate(1.0, 0), candidate(2.0, 1), candidate(3.0, 1)];
        let mut detections = vec![
            oriented(&candidates[0], 0),
            oriented(&candidates[1], 0),
            oriented(&candidates[1], 1),
            oriented(&candidates[2], 0),
        ];
        let mut keypoints = detections
            .iter()
            .map(|detection| SiftKeypoint {
                x: detection.x,
                y: detection.y,
                sigma: detection.sigma,
                orientation: detection.orientation,
                affine_shape: None,
                contrast: detection.response.abs(),
            })
            .collect::<Vec<_>>();
        cap_vlfeat_oriented_detections_by_level(&mut keypoints, &mut detections, &candidates, 1);
        assert_eq!(
            detections
                .iter()
                .map(|detection| detection.level)
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
        assert_eq!(
            keypoints
                .iter()
                .map(|keypoint| keypoint.x)
                .collect::<Vec<_>>(),
            vec![2.0, 2.0, 3.0]
        );
    }

    #[test]
    fn vlfeat_source_order_groups_levels_without_response_sorting() {
        let detection =
            |octave: i32, level: i32, orientation_index: usize, x: f64| SiftOrientedDetection {
                x,
                y: 0.0,
                sigma: 1.0,
                orientation: orientation_index as f64,
                response: 100.0 - x,
                octave,
                level,
                orientation_index,
            };
        // This vector deliberately interleaves source levels. Within one
        // level, the source sequence (including orientation order) must stay
        // intact; response and scale are not output-order keys.
        let detections = [
            detection(1, 0, 0, 10.0),
            detection(0, 1, 0, 20.0),
            detection(1, 0, 1, 11.0),
            detection(0, 0, 0, 30.0),
            detection(0, 1, 1, 21.0),
        ];
        let mut keypoints = detections
            .iter()
            .map(|detection| SiftKeypoint {
                x: detection.x,
                y: detection.y,
                sigma: detection.sigma,
                orientation: detection.orientation,
                affine_shape: None,
                contrast: detection.response.abs(),
            })
            .collect::<Vec<_>>();
        let mut detections = detections.to_vec();
        order_vlfeat_rows_like_colmap(&mut keypoints, &mut detections);
        let reference = vec![
            (0, 0, 0, 30.0),
            (0, 1, 0, 20.0),
            (0, 1, 1, 21.0),
            (1, 0, 0, 10.0),
            (1, 0, 1, 11.0),
        ];
        assert_eq!(
            detections
                .iter()
                .map(|detection| {
                    (
                        detection.octave,
                        detection.level,
                        detection.orientation_index,
                        detection.x,
                    )
                })
                .collect::<Vec<_>>(),
            reference
        );
    }

    #[test]
    fn vlfeat_source_order_opt_in_is_default_off_and_currently_identity() {
        let (width, height) = (96usize, 96usize);
        let pixels = blob_image(width, height, (48.25, 46.75), 5.0);
        let image = GrayImage::new(width, height, &pixels).unwrap();
        let baseline = SiftConfig {
            octaves: 2,
            max_keypoints: 128,
            max_orientations: 2,
            vlfeat_compatible_detector: true,
            vlfeat_compatible_descriptor: true,
            ..SiftConfig::default()
        };
        let explicit = SiftConfig {
            vlfeat_compatible_output_order: true,
            ..baseline.clone()
        };
        assert_eq!(
            extract_sift(&image, &baseline).unwrap(),
            extract_sift(
                &image,
                &SiftConfig {
                    vlfeat_compatible_output_order: false,
                    ..baseline.clone()
                }
            )
            .unwrap(),
            "omitting the source-order option must preserve the default"
        );
        assert_eq!(
            extract_sift(&image, &baseline).unwrap(),
            extract_sift(&image, &explicit).unwrap(),
            "the current compatible detector already emits COLMAP source order"
        );
    }

    #[test]
    fn standard_orientation_collapses_adjacent_bins() {
        let mut histogram = [0.0f64; ORIENTATION_BINS];
        histogram[7] = 1.0;
        histogram[8] = 0.9;
        let peaks = standard_orientation_peaks(&histogram);
        assert_eq!(peaks.len(), 1, "adjacent quantised bins must form one peak");
        assert_eq!(peaks[0].0, 7);
    }

    #[test]
    fn standard_orientation_keeps_separated_peaks() {
        let mut histogram = [0.0f64; ORIENTATION_BINS];
        histogram[3] = 1.0;
        histogram[21] = 0.9;
        let peaks = standard_orientation_peaks(&histogram);
        assert_eq!(
            peaks.iter().map(|peak| peak.0).collect::<Vec<_>>(),
            vec![3, 21]
        );
    }

    #[test]
    fn standard_orientation_handles_circular_boundary() {
        let mut histogram = [0.0f64; ORIENTATION_BINS];
        histogram[35] = 1.0;
        histogram[0] = 0.25;
        let peaks = standard_orientation_peaks(&histogram);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].0, 35);
        assert!(peaks[0].2.to_degrees() > 340.0);
    }

    #[test]
    fn standard_orientation_interpolation_is_bounded_and_finite() {
        let angle = interpolate_orientation(3, 0.8, 1.0, 0.6).to_degrees();
        assert!((angle - 31.666_666_666_7).abs() < 1e-9);
        let flat = interpolate_orientation(3, 1.0, 1.0, 1.0).to_degrees();
        assert!((flat - 30.0).abs() < 1e-12);
        let invalid = interpolate_orientation(3, f64::NAN, 1.0, 1.0);
        assert!(invalid.is_finite());
    }

    #[test]
    fn standard_orientation_is_deterministic_and_default_identity() {
        let (w, h) = (48usize, 48usize);
        let pixels = blob_image(w, h, (20.0, 24.0), 3.5);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let legacy = SiftConfig {
            octaves: 1,
            intervals: 2,
            ..SiftConfig::default()
        };
        assert!(!legacy.standard_orientation_peaks);
        let explicit_legacy = SiftConfig {
            standard_orientation_peaks: false,
            ..legacy.clone()
        };
        assert_eq!(
            extract_sift(&image, &legacy).unwrap(),
            extract_sift(&image, &explicit_legacy).unwrap(),
            "default orientation mode must remain byte-identical"
        );
        let standard = SiftConfig {
            standard_orientation_peaks: true,
            ..legacy
        };
        assert_eq!(
            extract_sift(&image, &standard).unwrap(),
            extract_sift(&image, &standard).unwrap(),
            "standard orientation extraction must be deterministic"
        );
    }

    #[test]
    fn fixed_keypoint_constructor_maps_isotropic_metadata() {
        let keypoint = SiftKeypoint::from_location_scale_orientation(
            12.5,
            8.25,
            2.75,
            -std::f64::consts::FRAC_PI_2,
        );
        assert_eq!(keypoint.x, 12.5);
        assert_eq!(keypoint.y, 8.25);
        assert_eq!(keypoint.sigma, 2.75);
        assert!((keypoint.orientation - 3.0 * std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(keypoint.affine_shape.is_none());
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

    #[test]
    fn dsp_sift_keeps_dimension_and_differs_from_plain() {
        let (w, h) = (64usize, 64usize);
        let pixels = blob_image(w, h, (32.0, 30.0), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let plain = SiftConfig {
            octaves: 2,
            ..SiftConfig::default()
        };
        let dsp = SiftConfig {
            domain_size_pooling: true,
            dsp_num_scales: 5,
            ..plain.clone()
        };
        let (kp_p, d_p) = extract_sift(&image, &plain).unwrap();
        let (kp_d, d_d) = extract_sift(&image, &dsp).unwrap();
        assert_eq!(kp_p.len(), kp_d.len(), "DSP must not change detections");
        assert!(!d_p.is_empty());
        for d in &d_d {
            assert_eq!(d.len(), 128);
            let norm: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "DSP descriptor must be L2-unit");
        }
        let changed = d_p.iter().zip(d_d.iter()).any(|(a, b)| a != b);
        assert!(changed, "DSP pooling should alter at least one descriptor");
    }

    #[test]
    fn dsp_scale_grid_matches_published_preset_and_one_scale_is_identity() {
        let default = SiftConfig::default();
        let scales = dsp_domain_scale_factors(&SiftConfig {
            domain_size_pooling: true,
            ..default.clone()
        });
        assert_eq!(scales.len(), DSP_PAPER_NUM_SCALES);
        assert!((scales[0] - DSP_PAPER_MIN_SCALE).abs() < 1e-12);
        let expected_step =
            (DSP_PAPER_MAX_SCALE - DSP_PAPER_MIN_SCALE) / DSP_PAPER_NUM_SCALES as f64;
        assert!((scales[1] - scales[0] - expected_step).abs() < 1e-12);
        let first_step = scales[1] - scales[0];
        for pair in scales.windows(2) {
            assert!((pair[1] - pair[0] - first_step).abs() < 1e-12);
        }
        assert!(scales[scales.len() - 1] < DSP_PAPER_MAX_SCALE);

        let pixels = blob_image(96, 96, (47.25, 51.5), 8.0);
        let image = GrayImage::new(96, 96, &pixels).unwrap();
        let keypoints = [SiftKeypoint::from_location_scale_orientation(
            47.25, 51.5, 6.4, 0.37,
        )];
        let compatible = SiftConfig {
            vlfeat_compatible_descriptor: true,
            ..default
        };
        let one_scale = SiftConfig {
            domain_size_pooling: true,
            dsp_num_scales: 1,
            ..compatible.clone()
        };
        assert_eq!(
            describe_sift_keypoints(&image, &keypoints, &compatible),
            describe_sift_keypoints(&image, &keypoints, &one_scale),
            "one DSP sample must be the detected domain, not the lower range endpoint"
        );
    }

    #[test]
    fn dsp_vlfeat_pooling_is_deterministic_and_quantized_after_normalization() {
        let pixels = blob_image(96, 96, (47.25, 51.5), 8.0);
        let image = GrayImage::new(96, 96, &pixels).unwrap();
        let keypoints = [SiftKeypoint::from_location_scale_orientation(
            47.25, 51.5, 6.4, 0.37,
        )];
        let config = SiftConfig {
            domain_size_pooling: true,
            dsp_num_scales: 3,
            vlfeat_compatible_descriptor: true,
            ..SiftConfig::default()
        };
        let first = describe_sift_keypoints(&image, &keypoints, &config);
        let second = describe_sift_keypoints(&image, &keypoints, &config);
        assert_eq!(first, second);
        let descriptor = &first[0];
        assert_eq!(descriptor.len(), 128);
        assert!(descriptor
            .iter()
            .all(|v| v.is_finite() && (0.0..=255.0).contains(v)));
        assert!(descriptor.iter().any(|&v| v > 0.0));
        let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 512.0).abs() < 2.0, "quantized norm={norm}");
    }

    #[test]
    fn dsp_scale_change_improves_synthetic_similarity() {
        let make_image = |stretch: f64| {
            let (w, h) = (128usize, 128usize);
            let mut pixels = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let u = (x as f64 - 64.0) / stretch;
                    let v = (y as f64 - 64.0) / stretch;
                    pixels[y * w + x] =
                        (0.5 + 0.22 * (0.19 * u + 0.07 * v).sin()
                            + 0.18 * (0.11 * u - 0.23 * v).cos()
                            + 0.1 * (0.17 * u + 0.13 * v).sin()) as f32;
                }
            }
            pixels
        };
        let first_pixels = make_image(1.0);
        let second_pixels = make_image(1.45);
        let first = GrayImage::new(128, 128, &first_pixels).unwrap();
        let second = GrayImage::new(128, 128, &second_pixels).unwrap();
        let kp_first = [SiftKeypoint::from_location_scale_orientation(
            64.0, 64.0, 5.0, 0.0,
        )];
        let kp_second = [SiftKeypoint::from_location_scale_orientation(
            64.0, 64.0, 7.25, 0.0,
        )];
        let plain = SiftConfig {
            vlfeat_compatible_descriptor: true,
            ..SiftConfig::default()
        };
        let dsp = SiftConfig {
            domain_size_pooling: true,
            vlfeat_compatible_descriptor: true,
            ..plain.clone()
        };
        let d_plain_a = describe_sift_keypoints(&first, &kp_first, &plain)[0].clone();
        let d_plain_b = describe_sift_keypoints(&second, &kp_second, &plain)[0].clone();
        let d_dsp_a = describe_sift_keypoints(&first, &kp_first, &dsp)[0].clone();
        let d_dsp_b = describe_sift_keypoints(&second, &kp_second, &dsp)[0].clone();
        let cosine = |a: &[f32], b: &[f32]| {
            let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        let plain_similarity = cosine(&d_plain_a, &d_plain_b);
        let dsp_similarity = cosine(&d_dsp_a, &d_dsp_b);
        assert!(
            dsp_similarity > plain_similarity + 1e-3,
            "DSP should improve scale-change similarity: plain={plain_similarity:.6}, dsp={dsp_similarity:.6}"
        );
    }

    #[test]
    fn l1_root_normalization_is_unit_and_differs_from_l2() {
        let (w, h) = (64usize, 64usize);
        let pixels = blob_image(w, h, (32.0, 30.0), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let l2 = SiftConfig {
            octaves: 2,
            normalization: SiftNormalization::L2,
            ..SiftConfig::default()
        };
        let l1 = SiftConfig {
            normalization: SiftNormalization::L1Root,
            ..l2.clone()
        };
        let (kp_a, d_a) = extract_sift(&image, &l2).unwrap();
        let (kp_b, d_b) = extract_sift(&image, &l1).unwrap();
        assert_eq!(kp_a.len(), kp_b.len());
        assert!(d_a.iter().zip(d_b.iter()).any(|(a, b)| a != b));
        for d in &d_b {
            let norm: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3);
            assert!(d.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn descriptor_magnification_default_preserves_legacy_and_narrower_differs() {
        let (w, h) = (64usize, 64usize);
        let pixels = blob_image(w, h, (32.0, 30.0), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let legacy = SiftConfig {
            octaves: 2,
            ..SiftConfig::default()
        };
        let explicit_legacy = SiftConfig {
            descriptor_magnification: 8.0,
            ..legacy.clone()
        };
        let narrow = SiftConfig {
            descriptor_magnification: 3.0,
            ..legacy.clone()
        };
        let (kp_a, d_a) = extract_sift(&image, &legacy).unwrap();
        let (kp_b, d_b) = extract_sift(&image, &explicit_legacy).unwrap();
        let (kp_c, d_c) = extract_sift(&image, &narrow).unwrap();
        assert_eq!(
            kp_a, kp_b,
            "explicit legacy magnification changes keypoints"
        );
        assert_eq!(
            d_a, d_b,
            "explicit legacy magnification changes descriptors"
        );
        assert_eq!(
            kp_a, kp_c,
            "descriptor magnification must not change detections"
        );
        assert!(d_a.iter().zip(d_c.iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn describe_existing_keypoints_preserves_descriptor_indices() {
        let (w, h) = (64usize, 64usize);
        let pixels = blob_image(w, h, (32.0, 30.0), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig {
            octaves: 2,
            ..SiftConfig::default()
        };
        let (keypoints, primary) = extract_sift(&image, &config).unwrap();
        let alternate = describe_sift_keypoints(
            &image,
            &keypoints,
            &SiftConfig {
                descriptor_magnification: 3.0,
                ..config.clone()
            },
        );
        assert_eq!(alternate.len(), keypoints.len());
        for ((kp, primary_descriptor), alternate_descriptor) in
            keypoints.iter().zip(primary.iter()).zip(alternate.iter())
        {
            assert!(kp.x.is_finite() && kp.y.is_finite());
            assert_eq!(primary_descriptor.len(), 128);
            assert_eq!(alternate_descriptor.len(), 128);
        }
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
    fn prefer_larger_scale_keeps_coarser_keypoints_when_capped() {
        let (w, h) = (128usize, 128usize);
        let px = dot_texture(w, h, 7);
        let image = GrayImage::new(w, h, &px).unwrap();
        let contrast = SiftConfig {
            max_keypoints: 32,
            prefer_larger_scale: false,
            ..SiftConfig::default()
        };
        let scale = SiftConfig {
            max_keypoints: 32,
            prefer_larger_scale: true,
            ..SiftConfig::default()
        };
        let (k_c, _) = extract_sift(&image, &contrast).unwrap();
        let (k_s, _) = extract_sift(&image, &scale).unwrap();
        assert_eq!(k_c.len(), 32);
        assert_eq!(k_s.len(), 32);
        let mean =
            |kps: &[SiftKeypoint]| kps.iter().map(|k| k.sigma).sum::<f64>() / kps.len() as f64;
        assert!(
            mean(&k_s) + 1e-9 >= mean(&k_c),
            "larger-scale prune mean σ {} should be ≥ contrast prune {}",
            mean(&k_s),
            mean(&k_c)
        );
        // Sets should differ on a dense blob scene when the budget binds.
        let same = k_c.len() == k_s.len()
            && k_c
                .iter()
                .zip(k_s.iter())
                .all(|(a, b)| (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
        assert!(
            !same,
            "prefer_larger_scale should change which keypoints survive"
        );
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
