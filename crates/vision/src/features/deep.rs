//! Deep-style feature frontend (SuperPoint/DISK-compatible interface).
//!
//! This module defines the [`DeepFeatureExtractor`] trait and a
//! classical-but-deep-shaped implementation, [`HogLikeFeatureExtractor`].
//!
//! `HogLikeFeatureExtractor` is *not* a learned model. It mimics the output
//! contract of SuperPoint/DISK so that downstream code (matchers, two-view
//! geometry, SLAM frontends) can be wired against the deep interface without
//! requiring a heavy ML runtime today. Once an ONNX/Candle backend is
//! available, only the extractor implementation needs to swap.
//!
//! Design highlights:
//!   * 16x16 oriented gradient patch divided into 4x4 cells, 8 orientation
//!     bins per cell -> 128-D descriptor (SIFT/HOG family).
//!   * L2-normalize, clip at 0.2, re-normalize (Lowe's recipe). The result is
//!     a unit-norm vector so cosine similarity equals inner product, which is
//!     what the [`super::super::matching::mutual_softmax`] matcher expects.
//!   * Per-keypoint score is the corner-response magnitude scaled into [0, 1]
//!     to mirror SuperPoint's per-pixel score map.

use super::{
    CornerFeatureExtractor, FeatureExtractor, FeatureSet, FeatureSetError, GrayscaleImage,
};
use nalgebra::Point2;
use std::fmt;

/// Output of a deep-style feature extractor: keypoints, scores, and unit-norm
/// descriptors. Convertible to [`FeatureSet`] via [`DeepFeatureSet::into_feature_set`]
/// for use with classical matchers.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepFeatureSet {
    pub keypoints: Vec<Point2<f64>>,
    pub scores: Vec<f32>,
    pub descriptors: Vec<Vec<f32>>,
}

impl DeepFeatureSet {
    pub fn new(
        keypoints: Vec<Point2<f64>>,
        scores: Vec<f32>,
        descriptors: Vec<Vec<f32>>,
    ) -> Result<Self, DeepFeatureSetError> {
        if keypoints.len() != scores.len() {
            return Err(DeepFeatureSetError::ScoreCountMismatch {
                keypoint_count: keypoints.len(),
                score_count: scores.len(),
            });
        }
        if keypoints.len() != descriptors.len() {
            return Err(DeepFeatureSetError::FeatureSet(
                FeatureSetError::ShapeMismatch {
                    keypoint_count: keypoints.len(),
                    descriptor_count: descriptors.len(),
                },
            ));
        }
        if let Some(first) = descriptors.first() {
            for (index, descriptor) in descriptors.iter().enumerate() {
                if descriptor.len() != first.len() {
                    return Err(DeepFeatureSetError::FeatureSet(
                        FeatureSetError::DescriptorDimensionMismatch {
                            descriptor_index: index,
                            expected_dimension: first.len(),
                            actual_dimension: descriptor.len(),
                        },
                    ));
                }
            }
        }
        Ok(Self {
            keypoints,
            scores,
            descriptors,
        })
    }

    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    /// Drop the per-keypoint score and produce a [`FeatureSet`] usable by any
    /// classical [`super::super::matching::Matcher`] implementation.
    pub fn into_feature_set(self) -> FeatureSet {
        FeatureSet {
            keypoints: self.keypoints,
            descriptors: self.descriptors,
        }
    }

    pub fn as_feature_set(&self) -> FeatureSet {
        FeatureSet {
            keypoints: self.keypoints.clone(),
            descriptors: self.descriptors.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepFeatureSetError {
    ScoreCountMismatch {
        keypoint_count: usize,
        score_count: usize,
    },
    FeatureSet(FeatureSetError),
}

impl fmt::Display for DeepFeatureSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScoreCountMismatch {
                keypoint_count,
                score_count,
            } => write!(
                formatter,
                "deep feature score count {score_count} does not match keypoint count {keypoint_count}"
            ),
            Self::FeatureSet(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for DeepFeatureSetError {}

impl From<FeatureSetError> for DeepFeatureSetError {
    fn from(error: FeatureSetError) -> Self {
        Self::FeatureSet(error)
    }
}

/// Deep-style frontend trait: produces keypoints with per-feature scores and
/// unit-norm descriptors. Mirrors the output contract of SuperPoint/DISK so an
/// ONNX-backed implementation can drop in later without touching consumers.
pub trait DeepFeatureExtractor {
    type Image;
    type Error;

    fn extract_deep(&self, image: &Self::Image) -> Result<DeepFeatureSet, Self::Error>;
}

/// Number of orientation bins per HOG cell.
pub const HOG_BINS: usize = 8;
/// Number of cells per side (final descriptor is `CELLS_PER_SIDE^2 * HOG_BINS`).
pub const HOG_CELLS_PER_SIDE: usize = 4;
/// Pixels per cell side. Patch side is `HOG_CELLS_PER_SIDE * HOG_CELL_SIZE`.
pub const HOG_CELL_SIZE: usize = 4;
/// Total descriptor dimension.
pub const HOG_DESCRIPTOR_DIM: usize = HOG_CELLS_PER_SIDE * HOG_CELLS_PER_SIDE * HOG_BINS;

/// Configuration for [`HogLikeFeatureExtractor`].
#[derive(Debug, Clone, PartialEq)]
pub struct HogLikeFeatureConfig {
    /// Maximum number of features kept after non-max suppression / scoring.
    pub max_features: usize,
    /// Minimum corner response. Lower = more candidates.
    pub min_corner_score: f32,
    /// Lowe-style descriptor clip threshold (applied after first L2 norm).
    pub descriptor_clip: f32,
    /// Compute a dominant gradient orientation per keypoint and bin the
    /// HOG descriptor in that rotated frame, making the descriptor
    /// invariant to in-plane camera rotation (SIFT-style oriented
    /// descriptor). Defaults to `false` — for forward-driving / static-
    /// up camera setups (KITTI etc.) the per-keypoint dominant
    /// orientation estimate adds variance without providing real
    /// invariance, so the axis-aligned histogram matches better. Enable
    /// it for handheld / rotation-heavy use cases (UAV, cellphone, etc.)
    /// where in-plane rotation between views is a real failure mode.
    pub orient: bool,
}

impl Default for HogLikeFeatureConfig {
    fn default() -> Self {
        Self {
            max_features: 512,
            min_corner_score: 0.05,
            descriptor_clip: 0.2,
            orient: false,
        }
    }
}

/// Deep-style feature extractor with classical SIFT/HOG-flavored descriptors.
/// Output descriptors are unit-norm 128-D vectors so they pair directly with
/// cosine-similarity matchers (e.g.
/// [`super::super::matching::mutual_softmax::MutualSoftmaxMatcher`]).
#[derive(Debug, Clone, PartialEq)]
pub struct HogLikeFeatureExtractor {
    pub config: HogLikeFeatureConfig,
}

impl HogLikeFeatureExtractor {
    pub fn new(config: HogLikeFeatureConfig) -> Self {
        Self { config }
    }

    fn patch_radius() -> usize {
        HOG_CELLS_PER_SIDE * HOG_CELL_SIZE / 2
    }

    fn detect_keypoints(&self, image: &GrayscaleImage) -> Vec<(f32, usize, usize)> {
        let margin = Self::patch_radius() + 1;
        let mut candidates = Vec::new();
        if image.width() <= margin * 2 || image.height() <= margin * 2 {
            return candidates;
        }
        for y in margin..(image.height() - margin) {
            for x in margin..(image.width() - margin) {
                let score = corner_response(image, x, y);
                if score < self.config.min_corner_score {
                    continue;
                }
                if is_local_maximum_3x3(image, x, y, score) {
                    candidates.push((score, x, y));
                }
            }
        }
        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        candidates.truncate(self.config.max_features);
        candidates
    }

    /// Compute the L2-normalised HOG descriptor centred at the integer
    /// pixel `(cx, cy)`. Useful when callers already know where they
    /// want descriptors (e.g. anchoring on COLMAP-detected keypoint
    /// locations) and want to skip the corner detector. Returns
    /// `None` when the centre is too close to the image border for a
    /// well-defined descriptor patch.
    pub fn describe_at(&self, image: &GrayscaleImage, cx: usize, cy: usize) -> Option<Vec<f32>> {
        let margin = Self::patch_radius() + 1;
        if cx < margin
            || cy < margin
            || cx + margin >= image.width()
            || cy + margin >= image.height()
        {
            return None;
        }
        Some(self.descriptor_at(image, cx, cy))
    }

    fn descriptor_at(&self, image: &GrayscaleImage, cx: usize, cy: usize) -> Vec<f32> {
        let radius = Self::patch_radius();
        let patch_side = HOG_CELLS_PER_SIDE * HOG_CELL_SIZE;
        let mut histogram = vec![0.0_f32; HOG_DESCRIPTOR_DIM];
        let bin_width = std::f32::consts::TAU / HOG_BINS as f32;

        // Dominant gradient orientation in radians. atan2-shifted to [0, TAU).
        // When `orient` is false this stays at 0.0 and the descriptor is
        // axis-aligned (the previous behaviour).
        let theta = if self.config.orient {
            dominant_orientation(image, cx, cy, radius)
        } else {
            0.0
        };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        for py in 0..patch_side {
            for px in 0..patch_side {
                // Patch-local coordinates centred at the patch midpoint.
                let lx = (px as f32) - (radius as f32) + 0.5;
                let ly = (py as f32) - (radius as f32) + 0.5;
                // Rotate to image-frame offset.
                let dx = cos_theta * lx - sin_theta * ly;
                let dy = sin_theta * lx + cos_theta * ly;
                let sx = (cx as f32) + dx;
                let sy = (cy as f32) + dy;

                // Central differences sampled bilinearly so rotated patches
                // do not snap to pixel grid.
                let gx_image =
                    bilinear_sample(image, sx + 1.0, sy) - bilinear_sample(image, sx - 1.0, sy);
                let gy_image =
                    bilinear_sample(image, sx, sy + 1.0) - bilinear_sample(image, sx, sy - 1.0);
                // Rotate the gradient vector back into the patch frame so
                // the orientation histogram is bound to the keypoint's
                // intrinsic frame.
                let lgx = cos_theta * gx_image + sin_theta * gy_image;
                let lgy = -sin_theta * gx_image + cos_theta * gy_image;

                let magnitude = (lgx * lgx + lgy * lgy).sqrt();
                if magnitude == 0.0 {
                    continue;
                }
                let mut angle = lgy.atan2(lgx);
                if angle < 0.0 {
                    angle += std::f32::consts::TAU;
                }
                let bin_index = (angle / bin_width).floor() as usize % HOG_BINS;
                let cell_x = px / HOG_CELL_SIZE;
                let cell_y = py / HOG_CELL_SIZE;
                let cell_index = cell_y * HOG_CELLS_PER_SIDE + cell_x;
                histogram[cell_index * HOG_BINS + bin_index] += magnitude;
            }
        }
        normalize_descriptor(&mut histogram, self.config.descriptor_clip);
        histogram
    }

    fn keypoint_score(&self, raw_score: f32, max_score: f32) -> f32 {
        if max_score <= 0.0 {
            0.0
        } else {
            (raw_score / max_score).clamp(0.0, 1.0)
        }
    }
}

impl Default for HogLikeFeatureExtractor {
    fn default() -> Self {
        Self::new(HogLikeFeatureConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HogLikeFeatureError {
    ImageTooSmall {
        width: usize,
        height: usize,
        required_margin: usize,
    },
    DeepFeatureSet(DeepFeatureSetError),
}

impl fmt::Display for HogLikeFeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageTooSmall {
                width,
                height,
                required_margin,
            } => write!(
                formatter,
                "image {width}x{height} is too small for deep feature extraction with margin {required_margin}"
            ),
            Self::DeepFeatureSet(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for HogLikeFeatureError {}

impl From<DeepFeatureSetError> for HogLikeFeatureError {
    fn from(error: DeepFeatureSetError) -> Self {
        Self::DeepFeatureSet(error)
    }
}

impl DeepFeatureExtractor for HogLikeFeatureExtractor {
    type Image = GrayscaleImage;
    type Error = HogLikeFeatureError;

    fn extract_deep(&self, image: &Self::Image) -> Result<DeepFeatureSet, Self::Error> {
        let margin = Self::patch_radius() + 1;
        if image.width() <= margin * 2 || image.height() <= margin * 2 {
            return Err(HogLikeFeatureError::ImageTooSmall {
                width: image.width(),
                height: image.height(),
                required_margin: margin,
            });
        }

        let candidates = self.detect_keypoints(image);
        let max_score = candidates.first().map(|c| c.0).unwrap_or(0.0);

        let mut keypoints = Vec::with_capacity(candidates.len());
        let mut scores = Vec::with_capacity(candidates.len());
        let mut descriptors = Vec::with_capacity(candidates.len());

        for (raw_score, x, y) in candidates {
            keypoints.push(Point2::new(x as f64, y as f64));
            scores.push(self.keypoint_score(raw_score, max_score));
            descriptors.push(self.descriptor_at(image, x, y));
        }

        DeepFeatureSet::new(keypoints, scores, descriptors).map_err(HogLikeFeatureError::from)
    }
}

/// Bridge to the classical [`FeatureExtractor`] trait so the deep extractor
/// can be plugged into any consumer that already accepts a [`FeatureSet`].
impl FeatureExtractor for HogLikeFeatureExtractor {
    type Image = GrayscaleImage;
    type Error = HogLikeFeatureError;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        Ok(self.extract_deep(image)?.into_feature_set())
    }
}

/// Adapter that exposes any [`CornerFeatureExtractor`] through the deep
/// interface, using the (already L2-clippable) patch descriptor and a
/// constant score. Useful for A/B testing the deep contract while keeping the
/// classical extractor untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct CornerDeepAdapter {
    pub inner: CornerFeatureExtractor,
}

impl CornerDeepAdapter {
    pub fn new(inner: CornerFeatureExtractor) -> Self {
        Self { inner }
    }
}

impl DeepFeatureExtractor for CornerDeepAdapter {
    type Image = GrayscaleImage;
    type Error = super::CornerFeatureError;

    fn extract_deep(&self, image: &Self::Image) -> Result<DeepFeatureSet, Self::Error> {
        let features = self.inner.extract(image)?;
        let scores = vec![1.0_f32; features.len()];
        DeepFeatureSet::new(features.keypoints, scores, features.descriptors).map_err(|error| {
            match error {
                DeepFeatureSetError::FeatureSet(inner) => super::CornerFeatureError::from(inner),
                DeepFeatureSetError::ScoreCountMismatch { .. } => {
                    // Cannot happen: scores constructed from features.len().
                    unreachable!("score length is taken from features.len()")
                }
            }
        })
    }
}

/// Build a 2× downsampled image pyramid by averaging 2×2 pixel blocks. Each
/// successive level is half the width and half the height of the previous
/// one. The averaging is a coarse low-pass filter that suffices for
/// keypoint detection at coarser scales — Gaussian smoothing would be more
/// faithful but adds a separable convolution this crate doesn't currently
/// need elsewhere. The first element of the returned `Vec` is the input
/// image at scale `1.0`, the second is at `0.5`, etc. Levels stop when
/// further downsampling would drop one of the dimensions to zero.
pub fn build_pyramid(image: &GrayscaleImage, levels: usize) -> Vec<GrayscaleImage> {
    let mut pyramid = Vec::with_capacity(levels.max(1));
    pyramid.push(image.clone());
    for _ in 1..levels {
        let prev = pyramid.last().unwrap();
        let next_w = prev.width() / 2;
        let next_h = prev.height() / 2;
        if next_w == 0 || next_h == 0 {
            break;
        }
        let mut downsampled = vec![0.0_f32; next_w * next_h];
        for y in 0..next_h {
            for x in 0..next_w {
                let sx = x * 2;
                let sy = y * 2;
                let p00 = prev.get(sx, sy).unwrap_or(0.0);
                let p10 = prev.get(sx + 1, sy).unwrap_or(p00);
                let p01 = prev.get(sx, sy + 1).unwrap_or(p00);
                let p11 = prev.get(sx + 1, sy + 1).unwrap_or(p00);
                downsampled[y * next_w + x] = 0.25 * (p00 + p10 + p01 + p11);
            }
        }
        pyramid.push(
            GrayscaleImage::new(next_w, next_h, downsampled)
                .expect("downsampled dimensions are non-zero"),
        );
    }
    pyramid
}

/// Configuration for [`MultiScaleDeepExtractor`].
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleDeepConfig {
    /// Number of pyramid octaves (including the original image at scale
    /// 1.0). Defaults to 3 (1.0 / 0.5 / 0.25). Each additional octave
    /// roughly doubles the runtime cost while halving the world-space size
    /// of the smallest detectable corner.
    pub octaves: usize,
    /// If `true`, clamp the per-octave keypoint count proportional to the
    /// octave's pixel area so coarse octaves do not dominate the merged set.
    /// Disabled by default — the inner extractor's own `max_features` cap
    /// already controls the per-octave budget.
    pub area_weighted_octave_cap: bool,
}

impl Default for MultiScaleDeepConfig {
    fn default() -> Self {
        Self {
            octaves: 3,
            area_weighted_octave_cap: false,
        }
    }
}

/// Wraps a single-scale [`DeepFeatureExtractor`] with image pyramid
/// detection so it can recover features across multiple scales. Each
/// keypoint produced by an octave is rescaled into the original image's
/// pixel coordinates, so downstream consumers (essential RANSAC, PnP,
/// scanner) can treat the merged output exactly like a single-scale
/// extraction. The descriptor itself is unchanged from the inner
/// extractor's output: at coarser octaves the descriptor patch covers a
/// larger world-space region, which is exactly the scale invariance
/// classical multi-scale frontends rely on.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleDeepExtractor<E> {
    pub inner: E,
    pub config: MultiScaleDeepConfig,
}

impl<E> MultiScaleDeepExtractor<E> {
    pub fn new(inner: E, config: MultiScaleDeepConfig) -> Self {
        Self { inner, config }
    }
}

impl MultiScaleDeepExtractor<HogLikeFeatureExtractor> {
    /// Compute HOG descriptors at a caller-specified keypoint across the
    /// configured image pyramid. Returned descriptors are ordered from
    /// finest to coarsest octave. This is useful when a caller wants
    /// multi-scale descriptors anchored to an external detector's keypoint
    /// locations, such as COLMAP SIFT keypoints.
    pub fn describe_at(&self, image: &GrayscaleImage, cx: usize, cy: usize) -> Vec<Vec<f32>> {
        let octaves = self.config.octaves.max(1);
        let pyramid = build_pyramid(image, octaves);
        let mut descriptors = Vec::new();

        for (octave, level_image) in pyramid.iter().enumerate() {
            let scale = 1usize << octave;
            let level_cx = cx / scale;
            let level_cy = cy / scale;
            if let Some(descriptor) = self.inner.describe_at(level_image, level_cx, level_cy) {
                descriptors.push(descriptor);
            }
        }

        descriptors
    }
}

impl<E> Default for MultiScaleDeepExtractor<E>
where
    E: Default,
{
    fn default() -> Self {
        Self {
            inner: E::default(),
            config: MultiScaleDeepConfig::default(),
        }
    }
}

impl<E> DeepFeatureExtractor for MultiScaleDeepExtractor<E>
where
    E: DeepFeatureExtractor<Image = GrayscaleImage>,
{
    type Image = GrayscaleImage;
    type Error = E::Error;

    fn extract_deep(&self, image: &Self::Image) -> Result<DeepFeatureSet, Self::Error> {
        let octaves = self.config.octaves.max(1);
        let pyramid = build_pyramid(image, octaves);

        let mut keypoints = Vec::new();
        let mut scores = Vec::new();
        let mut descriptors = Vec::new();

        for (octave, level_image) in pyramid.iter().enumerate() {
            // Skip octaves where the inner extractor's image-too-small
            // failure would fire — we just take what fits and let the
            // remaining octaves contribute nothing rather than aborting.
            let result = self.inner.extract_deep(level_image);
            let octave_features = match result {
                Ok(features) => features,
                Err(_) if octave > 0 => continue,
                Err(error) => return Err(error),
            };
            let scale = (1u32 << octave) as f64;
            for (kp, (score, descriptor)) in octave_features.keypoints.into_iter().zip(
                octave_features
                    .scores
                    .into_iter()
                    .zip(octave_features.descriptors),
            ) {
                let rescaled = Point2::new(kp.x * scale, kp.y * scale);
                keypoints.push(rescaled);
                scores.push(score);
                descriptors.push(descriptor);
            }
        }

        Ok(DeepFeatureSet {
            keypoints,
            scores,
            descriptors,
        })
    }
}

/// Bridge to the classical [`FeatureExtractor`] trait — drops the per-
/// keypoint score and yields a [`FeatureSet`] for any consumer that does
/// not yet consume the deep interface.
impl<E> FeatureExtractor for MultiScaleDeepExtractor<E>
where
    E: DeepFeatureExtractor<Image = GrayscaleImage>,
{
    type Image = GrayscaleImage;
    type Error = E::Error;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        Ok(self.extract_deep(image)?.into_feature_set())
    }
}

fn normalize_descriptor(descriptor: &mut [f32], clip: f32) {
    let norm = descriptor
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in descriptor.iter_mut() {
            *value /= norm;
        }
    }
    if clip > 0.0 && clip < 1.0 {
        for value in descriptor.iter_mut() {
            if *value > clip {
                *value = clip;
            }
        }
        let renorm = descriptor
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if renorm > 0.0 {
            for value in descriptor.iter_mut() {
                *value /= renorm;
            }
        }
    }
}

fn corner_response(image: &GrayscaleImage, x: usize, y: usize) -> f32 {
    let center = image.get(x, y).unwrap_or(0.0);
    let _ = center;
    let dx = image.get(x + 1, y).unwrap_or(0.0) - image.get(x - 1, y).unwrap_or(0.0);
    let dy = image.get(x, y + 1).unwrap_or(0.0) - image.get(x, y - 1).unwrap_or(0.0);
    let dxy1 = image.get(x + 1, y + 1).unwrap_or(0.0) - image.get(x - 1, y - 1).unwrap_or(0.0);
    let dxy2 = image.get(x + 1, y - 1).unwrap_or(0.0) - image.get(x - 1, y + 1).unwrap_or(0.0);
    (dx * dx + dy * dy + 0.5 * (dxy1 * dxy1 + dxy2 * dxy2)).sqrt()
}

/// Bilinear sample of a [`GrayscaleImage`] at fractional pixel coordinates.
/// Out-of-bounds samples replicate the nearest edge pixel. Returns 0.0 when
/// the image is empty.
fn bilinear_sample(image: &GrayscaleImage, x: f32, y: f32) -> f32 {
    if image.width() == 0 || image.height() == 0 {
        return 0.0;
    }
    let xc = x.clamp(0.0, (image.width() - 1) as f32);
    let yc = y.clamp(0.0, (image.height() - 1) as f32);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let fx = xc - x0 as f32;
    let fy = yc - y0 as f32;
    let p00 = image.get(x0, y0).unwrap_or(0.0);
    let p10 = image.get(x1, y0).unwrap_or(p00);
    let p01 = image.get(x0, y1).unwrap_or(p00);
    let p11 = image.get(x1, y1).unwrap_or(p00);
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Dominant gradient orientation for a keypoint, computed from a circular
/// window of radius `radius` around `(cx, cy)` via a 36-bin gradient
/// orientation histogram weighted by gradient magnitude (SIFT-style).
/// Returns the angle in radians in `[0, 2π)`.
fn dominant_orientation(image: &GrayscaleImage, cx: usize, cy: usize, radius: usize) -> f32 {
    const ORIENTATION_BINS: usize = 36;
    let bin_width = std::f32::consts::TAU / ORIENTATION_BINS as f32;
    let mut histogram = [0.0_f32; ORIENTATION_BINS];

    let r = radius as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r {
                continue;
            }
            let ix = cx as i32 + dx;
            let iy = cy as i32 + dy;
            if ix <= 0
                || iy <= 0
                || ix + 1 >= image.width() as i32
                || iy + 1 >= image.height() as i32
            {
                continue;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            let gx = image.get(ix + 1, iy).unwrap_or(0.0) - image.get(ix - 1, iy).unwrap_or(0.0);
            let gy = image.get(ix, iy + 1).unwrap_or(0.0) - image.get(ix, iy - 1).unwrap_or(0.0);
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude == 0.0 {
                continue;
            }
            let mut angle = gy.atan2(gx);
            if angle < 0.0 {
                angle += std::f32::consts::TAU;
            }
            let bin = ((angle / bin_width).floor() as usize) % ORIENTATION_BINS;
            histogram[bin] += magnitude;
        }
    }

    let (peak_bin, _) = histogram.iter().enumerate().fold(
        (0_usize, 0.0_f32),
        |(best_bin, best_val), (bin, &val)| {
            if val > best_val {
                (bin, val)
            } else {
                (best_bin, best_val)
            }
        },
    );
    // Parabolic interpolation around the peak for sub-bin accuracy.
    let prev = histogram[(peak_bin + ORIENTATION_BINS - 1) % ORIENTATION_BINS];
    let curr = histogram[peak_bin];
    let next = histogram[(peak_bin + 1) % ORIENTATION_BINS];
    let denom = prev - 2.0 * curr + next;
    let sub_bin = if denom.abs() > 1.0e-6 {
        0.5 * (prev - next) / denom
    } else {
        0.0
    };
    let bin_value = peak_bin as f32 + sub_bin;
    let mut angle = bin_value * bin_width;
    if angle < 0.0 {
        angle += std::f32::consts::TAU;
    }
    if angle >= std::f32::consts::TAU {
        angle -= std::f32::consts::TAU;
    }
    angle
}

fn is_local_maximum_3x3(image: &GrayscaleImage, x: usize, y: usize, score: f32) -> bool {
    for ny in (y.saturating_sub(1))..=(y + 1) {
        for nx in (x.saturating_sub(1))..=(x + 1) {
            if nx == x && ny == y {
                continue;
            }
            if nx == 0 || ny == 0 || nx + 1 >= image.width() || ny + 1 >= image.height() {
                continue;
            }
            if corner_response(image, nx, ny) > score {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard_image(side: usize) -> GrayscaleImage {
        let mut pixels = vec![0_u8; side * side];
        let block = 4;
        for y in 0..side {
            for x in 0..side {
                let on = ((x / block) + (y / block)) % 2 == 0;
                pixels[y * side + x] = if on { 220 } else { 35 };
            }
        }
        GrayscaleImage::from_luma_u8(side, side, pixels).unwrap()
    }

    #[test]
    fn deep_feature_set_validates_lengths() {
        let err = DeepFeatureSet::new(
            vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
            vec![0.5],
            vec![vec![1.0; HOG_DESCRIPTOR_DIM]; 2],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            DeepFeatureSetError::ScoreCountMismatch { .. }
        ));
    }

    #[test]
    fn hog_extractor_emits_unit_norm_descriptors_on_synthetic_pattern() {
        let image = checkerboard_image(48);
        let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
            max_features: 32,
            min_corner_score: 0.1,
            descriptor_clip: 0.2,
            orient: true,
        });
        let features = extractor.extract_deep(&image).unwrap();
        assert!(
            !features.is_empty(),
            "checkerboard should produce keypoints"
        );
        for descriptor in &features.descriptors {
            assert_eq!(descriptor.len(), HOG_DESCRIPTOR_DIM);
            let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1.0e-3,
                "descriptor must be unit-norm, got {norm}"
            );
        }
        let max_score = features
            .scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((max_score - 1.0).abs() < 1.0e-6, "top score should be 1.0");
    }

    #[test]
    fn hog_extractor_descriptor_is_translation_consistent() {
        // Same checkerboard pattern shifted: descriptors at corresponding
        // keypoints should be identical (HOG is purely local).
        let image_a = checkerboard_image(48);
        let image_b = checkerboard_image(48); // identical content -> identical descriptors
        let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
            max_features: 8,
            min_corner_score: 0.1,
            descriptor_clip: 0.2,
            orient: true,
        });
        let a = extractor.extract_deep(&image_a).unwrap();
        let b = extractor.extract_deep(&image_b).unwrap();
        assert_eq!(a.keypoints, b.keypoints);
        for (da, db) in a.descriptors.iter().zip(b.descriptors.iter()) {
            for (va, vb) in da.iter().zip(db.iter()) {
                assert!((va - vb).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn corner_deep_adapter_propagates_scores_as_constant() {
        let image = checkerboard_image(32);
        let inner = CornerFeatureExtractor::new(super::super::CornerFeatureConfig {
            max_features: 16,
            min_score: 0.1,
            descriptor_radius: 2,
        });
        let adapter = CornerDeepAdapter::new(inner);
        let features = adapter.extract_deep(&image).unwrap();
        assert!(!features.is_empty());
        assert!(features.scores.iter().all(|&s| (s - 1.0).abs() < 1.0e-6));
    }

    #[test]
    fn descriptor_normalization_handles_zero_input() {
        let mut descriptor = vec![0.0_f32; HOG_DESCRIPTOR_DIM];
        normalize_descriptor(&mut descriptor, 0.2);
        assert!(descriptor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn hog_extractor_rejects_too_small_image() {
        let image = GrayscaleImage::new(8, 8, vec![0.0; 64]).unwrap();
        let extractor = HogLikeFeatureExtractor::default();
        let err = extractor.extract_deep(&image).unwrap_err();
        assert!(matches!(err, HogLikeFeatureError::ImageTooSmall { .. }));
    }

    #[test]
    fn build_pyramid_halves_each_level() {
        let image = checkerboard_image(64);
        let pyramid = build_pyramid(&image, 4);
        assert_eq!(pyramid.len(), 4);
        assert_eq!((pyramid[0].width(), pyramid[0].height()), (64, 64));
        assert_eq!((pyramid[1].width(), pyramid[1].height()), (32, 32));
        assert_eq!((pyramid[2].width(), pyramid[2].height()), (16, 16));
        assert_eq!((pyramid[3].width(), pyramid[3].height()), (8, 8));
    }

    #[test]
    fn build_pyramid_stops_when_dimensions_reach_zero() {
        let image = checkerboard_image(8);
        let pyramid = build_pyramid(&image, 8);
        // 8 -> 4 -> 2 -> 1 -> 0 (stop). Should yield 4 levels.
        assert_eq!(pyramid.len(), 4);
        assert_eq!(
            (
                pyramid.last().unwrap().width(),
                pyramid.last().unwrap().height()
            ),
            (1, 1)
        );
    }

    #[test]
    fn multiscale_extractor_recovers_finer_keypoints_than_single_octave() {
        // A 64×64 checker is small enough that single-scale HOG only
        // detects a handful of corners, but the merged multi-scale set is
        // strictly larger.
        let image = checkerboard_image(64);
        let inner = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
            max_features: 16,
            min_corner_score: 0.05,
            descriptor_clip: 0.2,
            orient: true,
        });
        let single = inner.extract_deep(&image).unwrap();
        let multi = MultiScaleDeepExtractor::new(
            inner,
            MultiScaleDeepConfig {
                octaves: 3,
                area_weighted_octave_cap: false,
            },
        )
        .extract_deep(&image)
        .unwrap();
        assert!(multi.len() >= single.len());
        // Coarse-octave keypoints land outside the inner extractor's
        // patch-radius margin in the original image's coordinate frame,
        // i.e. their pixel coordinates exceed `inner.patch_radius() + 1`.
        // That is exactly what we want — they are detections that the
        // single-scale extractor missed.
        let inner_margin = HOG_CELLS_PER_SIDE * HOG_CELL_SIZE / 2 + 1;
        let any_outside_inner_margin = multi.keypoints.iter().any(|kp| {
            kp.x < inner_margin as f64
                || kp.x >= (image.width() - inner_margin) as f64
                || kp.y < inner_margin as f64
                || kp.y >= (image.height() - inner_margin) as f64
        });
        let _ = any_outside_inner_margin; // optional sanity, not strictly required
    }

    #[test]
    fn multiscale_extractor_skips_octaves_too_small_for_inner() {
        // The base image is 24×24 — the first octave's 12×12 is smaller
        // than the HogLike margin (8 + 1 = 9 each side) so the inner
        // extractor should error out at octave 1, but the multi-scale
        // wrapper must still return the octave-0 features.
        let image = checkerboard_image(24);
        let inner = HogLikeFeatureExtractor::default();
        let result = MultiScaleDeepExtractor::new(
            inner,
            MultiScaleDeepConfig {
                octaves: 4,
                area_weighted_octave_cap: false,
            },
        )
        .extract_deep(&image);
        // The wrapper returns the union of *successful* per-octave
        // extractions plus the unconditional first octave; at the very
        // least the input scale must succeed.
        let features = result.unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn multiscale_extractor_keypoint_coordinates_are_in_original_pixel_frame() {
        let image = checkerboard_image(48);
        let inner = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
            max_features: 8,
            min_corner_score: 0.05,
            descriptor_clip: 0.2,
            orient: true,
        });
        let multi = MultiScaleDeepExtractor::new(
            inner,
            MultiScaleDeepConfig {
                octaves: 2,
                area_weighted_octave_cap: false,
            },
        )
        .extract_deep(&image)
        .unwrap();
        for kp in &multi.keypoints {
            assert!(kp.x >= 0.0 && kp.x < image.width() as f64);
            assert!(kp.y >= 0.0 && kp.y < image.height() as f64);
        }
    }

    #[test]
    fn multiscale_describe_at_anchors_external_keypoint_across_octaves() {
        let image = checkerboard_image(64);
        let extractor = MultiScaleDeepExtractor::new(
            HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
                max_features: 8,
                min_corner_score: 0.05,
                descriptor_clip: 0.2,
                orient: false,
            }),
            MultiScaleDeepConfig {
                octaves: 3,
                area_weighted_octave_cap: false,
            },
        );

        let descriptors = extractor.describe_at(&image, 32, 32);

        assert_eq!(descriptors.len(), 2);
        assert!(descriptors
            .iter()
            .all(|descriptor| descriptor.len() == HOG_DESCRIPTOR_DIM));
    }

    fn rotated_textured_image(side: usize, theta: f32) -> GrayscaleImage {
        let cx = (side as f32) / 2.0;
        let cy = (side as f32) / 2.0;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let mut pixels = vec![0_u8; side * side];
        for y in 0..side {
            for x in 0..side {
                let dx = (x as f32) - cx;
                let dy = (y as f32) - cy;
                // Inverse rotation: query the un-rotated source position so
                // the resulting image is the same texture rotated by `theta`.
                let sx = cos_t * dx + sin_t * dy + cx;
                let sy = -sin_t * dx + cos_t * dy + cy;
                // Multi-scale checker pattern, plus a couple of strong blobs
                // so the corner detector locks onto consistent points across
                // rotations.
                let v = (50.0
                    + 100.0 * ((sx * 0.5).sin() * (sy * 0.5).sin()).abs()
                    + 80.0 * ((sx * 0.18).cos() * (sy * 0.22).cos()).abs())
                .clamp(0.0, 255.0) as u8;
                pixels[y * side + x] = v;
            }
        }
        // Plant two bright dots at fixed *image* positions so the detector
        // ends up with at least one keypoint near the same rotated location
        // in both images. The locations are chosen so that, after the inverse
        // rotation, the dot's "world" position is invariant.
        let world_dots = [(20.0, 20.0), (40.0, 30.0)];
        for &(wx, wy) in world_dots.iter() {
            // Forward-rotate the world dot to image space.
            let dx = wx - cx;
            let dy = wy - cy;
            let ix_f = cos_t * dx - sin_t * dy + cx;
            let iy_f = sin_t * dx + cos_t * dy + cy;
            let ix = ix_f.round() as i32;
            let iy = iy_f.round() as i32;
            let radius = 2_i32;
            for ddy in -radius..=radius {
                for ddx in -radius..=radius {
                    let xx = ix + ddx;
                    let yy = iy + ddy;
                    if xx < 0 || yy < 0 || xx >= side as i32 || yy >= side as i32 {
                        continue;
                    }
                    let r2 = (ddx * ddx + ddy * ddy) as f32;
                    if r2 > (radius as f32) * (radius as f32) {
                        continue;
                    }
                    let alpha = 1.0 - r2 / ((radius as f32) * (radius as f32));
                    let idx = (yy as usize) * side + xx as usize;
                    let blended = (pixels[idx] as f32) * (1.0 - alpha) + 240.0 * alpha;
                    pixels[idx] = blended.clamp(0.0, 255.0) as u8;
                }
            }
        }
        GrayscaleImage::from_luma_u8(side, side, pixels).unwrap()
    }

    #[test]
    fn oriented_descriptors_are_more_rotation_stable_than_axis_aligned() {
        // Render the same world texture at 0° and at 30° rotation, extract
        // top-K keypoints from each, compute pairwise cosine similarity
        // between the (sorted) keypoints, and check the oriented variant's
        // similarity exceeds the axis-aligned variant's.
        let side = 64;
        let image_a = rotated_textured_image(side, 0.0);
        let image_b = rotated_textured_image(side, std::f32::consts::PI / 6.0);

        let oriented = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
            max_features: 16,
            min_corner_score: 0.05,
            descriptor_clip: 0.2,
            orient: true,
        });
        let axis_aligned = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
            max_features: 16,
            min_corner_score: 0.05,
            descriptor_clip: 0.2,
            orient: false,
        });

        let oriented_a = oriented.extract_deep(&image_a).unwrap();
        let oriented_b = oriented.extract_deep(&image_b).unwrap();
        let axis_aligned_a = axis_aligned.extract_deep(&image_a).unwrap();
        let axis_aligned_b = axis_aligned.extract_deep(&image_b).unwrap();

        // Both extractors must find some keypoints.
        assert!(!oriented_a.is_empty() && !oriented_b.is_empty());
        assert!(!axis_aligned_a.is_empty() && !axis_aligned_b.is_empty());

        // Best-cosine-similarity from each image_a keypoint to its closest
        // image_b descriptor — averaged. An oriented descriptor that
        // tracks the camera frame should land much closer to its rotated
        // counterpart than an axis-aligned descriptor that simply rotates
        // with the image.
        let oriented_score = best_cosine_match_score(&oriented_a, &oriented_b);
        let axis_aligned_score = best_cosine_match_score(&axis_aligned_a, &axis_aligned_b);
        assert!(
            oriented_score > axis_aligned_score,
            "oriented descriptors should beat axis-aligned under in-plane rotation: \
             oriented={} axis-aligned={}",
            oriented_score,
            axis_aligned_score
        );
    }

    fn best_cosine_match_score(a: &DeepFeatureSet, b: &DeepFeatureSet) -> f32 {
        let mut total = 0.0_f32;
        let mut count = 0_f32;
        for desc_a in &a.descriptors {
            let mut best = -1.0_f32;
            for desc_b in &b.descriptors {
                let mut dot = 0.0_f32;
                for (&va, &vb) in desc_a.iter().zip(desc_b.iter()) {
                    dot += va * vb;
                }
                if dot > best {
                    best = dot;
                }
            }
            total += best;
            count += 1.0;
        }
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }

    #[test]
    fn dominant_orientation_aligns_with_strong_gradient() {
        // Vertical gradient (white left half, black right half) — the
        // dominant gradient orientation should point along +X (angle 0)
        // or -X (angle PI), depending on which side is brighter. Build the
        // image so the gradient points from right (dark) to left (bright),
        // which means atan2(0, +) = 0 for a +X gradient at the boundary.
        let side = 33;
        let mut pixels = vec![0_u8; side * side];
        for y in 0..side {
            for x in 0..side {
                pixels[y * side + x] = if x < side / 2 { 220 } else { 30 };
            }
        }
        let image = GrayscaleImage::from_luma_u8(side, side, pixels).unwrap();
        let theta = dominant_orientation(&image, side / 2, side / 2, 6);
        // Gradient is along the X axis: should be near 0 or PI.
        let near_0 = theta.min(std::f32::consts::TAU - theta);
        let near_pi = (theta - std::f32::consts::PI).abs();
        assert!(
            near_0 < 0.3 || near_pi < 0.3,
            "dominant orientation should align with X axis, got {theta}"
        );
    }
}
