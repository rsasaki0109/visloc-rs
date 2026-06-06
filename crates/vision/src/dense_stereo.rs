//! Dense rectified-stereo matching: a per-pixel disparity map by block matching,
//! and its back-projection to a dense metric point cloud.
//!
//! The sparse stereo path ([`crate::stereo`]) triangulates only feature
//! correspondences. This module densifies that: for *every* left pixel it
//! searches the same rectified row in the right image for the best-matching
//! block (sum-of-absolute-differences), with a left-right consistency check and
//! a uniqueness ratio to reject ambiguous matches. Feeding the surviving
//! per-pixel disparities through [`crate::stereo::triangulate_stereo_pixel`]
//! yields a dense metric point cloud per frame, which a caller fuses across
//! keyframes into a dense reconstruction.

use crate::features::GrayscaleImage;
use crate::stereo::triangulate_stereo_pixel;
use nalgebra::Point3;
use visloc_core::types::Camera;

/// Configuration for [`dense_disparity_map`] / [`dense_stereo_points`].
#[derive(Debug, Clone, PartialEq)]
pub struct DenseStereoConfig {
    /// Minimum disparity (px) considered. Larger ⇒ nearer max depth.
    pub min_disparity: usize,
    /// Maximum disparity (px) considered. Larger ⇒ nearer min depth, slower.
    pub max_disparity: usize,
    /// Half-window size of the SAD block (window is `2*block_radius+1` square).
    pub block_radius: usize,
    /// Reject a pixel whose best mean-absolute block difference (intensities in
    /// `[0,1]`) exceeds this — i.e. no good photometric match.
    pub max_block_diff: f32,
    /// Uniqueness: accept only if the best cost is at most this fraction of the
    /// second-best (distinct) cost. Lower ⇒ stricter. `1.0` disables it.
    pub uniqueness_ratio: f32,
    /// Left-right consistency tolerance (px). The right→left re-match must agree
    /// with the left→right disparity within this. `None` disables the check.
    pub lr_consistency_px: Option<f32>,
}

impl Default for DenseStereoConfig {
    fn default() -> Self {
        Self {
            min_disparity: 1,
            max_disparity: 96,
            block_radius: 3,
            max_block_diff: 0.08,
            uniqueness_ratio: 0.9,
            lr_consistency_px: Some(1.5),
        }
    }
}

/// A back-projected dense stereo point: the left-image pixel that produced it,
/// its left-camera-frame 3D position, and its intensity (for colouring).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseStereoPoint {
    /// Left-image pixel coordinate `(u, v)`.
    pub pixel: (usize, usize),
    /// Position in the left-camera frame (metres).
    pub point_cam: Point3<f64>,
    /// Disparity (px) used for the triangulation.
    pub disparity: f32,
    /// Left-image intensity in `[0, 1]` at the pixel (grayscale colour).
    pub intensity: f32,
}

#[inline]
fn at(img: &GrayscaleImage, x: usize, y: usize) -> f32 {
    img.pixels()[y * img.width() + x]
}

/// Mean absolute block difference between left block centred at `(lx, ly)` and
/// right block centred at `(rx, ly)` (same row — rectified). Returns `None` if
/// either window falls outside its image.
#[inline]
fn block_cost(
    left: &GrayscaleImage,
    right: &GrayscaleImage,
    lx: usize,
    rx: usize,
    ly: usize,
    r: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let n = (2 * r + 1) * (2 * r + 1);
    for dy in 0..=(2 * r) {
        let y = ly + dy - r;
        for dx in 0..=(2 * r) {
            sum += (at(left, lx + dx - r, y) - at(right, rx + dx - r, y)).abs();
        }
    }
    sum / n as f32
}

/// Search the rectified row for the best-matching disparity at left pixel
/// `(x, y)`. Returns `(best_disparity, best_cost, second_best_cost)`.
fn best_disparity_at(
    left: &GrayscaleImage,
    right: &GrayscaleImage,
    x: usize,
    y: usize,
    cfg: &DenseStereoConfig,
) -> Option<(usize, f32, f32)> {
    let r = cfg.block_radius;
    let mut best_d = 0usize;
    let mut best = f32::INFINITY;
    let mut second = f32::INFINITY;
    let d_lo = cfg.min_disparity.max(1);
    for d in d_lo..=cfg.max_disparity {
        // right pixel is shifted left by the disparity; keep the right block in-bounds
        if x < d + r {
            break;
        }
        let rx = x - d;
        let cost = block_cost(left, right, x, rx, y, r);
        if cost < best {
            second = best;
            best = cost;
            best_d = d;
        } else if cost < second {
            second = cost;
        }
    }
    if best.is_finite() {
        Some((best_d, best, second))
    } else {
        None
    }
}

/// Compute a per-left-pixel disparity map by block matching.
///
/// Returns a `width*height` row-major vector; invalid/rejected pixels are
/// `f32::NAN`. Both images must share dimensions.
pub fn dense_disparity_map(
    left: &GrayscaleImage,
    right: &GrayscaleImage,
    cfg: &DenseStereoConfig,
) -> Vec<f32> {
    let w = left.width();
    let h = left.height();
    let mut out = vec![f32::NAN; w * h];
    if right.width() != w || right.height() != h {
        return out;
    }
    let r = cfg.block_radius;
    if w <= 2 * r + cfg.max_disparity || h <= 2 * r {
        return out;
    }
    for y in r..(h - r) {
        for x in (r + cfg.min_disparity.max(1))..(w - r) {
            let Some((d, best, second)) = best_disparity_at(left, right, x, y, cfg) else {
                continue;
            };
            if best > cfg.max_block_diff {
                continue;
            }
            // Uniqueness: best must clearly beat the runner-up.
            if cfg.uniqueness_ratio < 1.0
                && second.is_finite()
                && best > cfg.uniqueness_ratio * second
            {
                continue;
            }
            // Left-right consistency: re-match this right pixel back to the left.
            if let Some(tol) = cfg.lr_consistency_px {
                let rx = x - d;
                if let Some((d_back, _, _)) = best_disparity_right(left, right, rx, y, cfg) {
                    if ((d as f32) - (d_back as f32)).abs() > tol {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            out[y * w + x] = d as f32;
        }
    }
    out
}

/// Right→left disparity search (the consistency check's reverse direction):
/// for right pixel `(rx, y)` find the left column `rx + d` that best matches.
fn best_disparity_right(
    left: &GrayscaleImage,
    right: &GrayscaleImage,
    rx: usize,
    y: usize,
    cfg: &DenseStereoConfig,
) -> Option<(usize, f32, f32)> {
    let r = cfg.block_radius;
    let w = left.width();
    let mut best_d = 0usize;
    let mut best = f32::INFINITY;
    let mut second = f32::INFINITY;
    let d_lo = cfg.min_disparity.max(1);
    for d in d_lo..=cfg.max_disparity {
        let lx = rx + d;
        if lx + r >= w {
            break;
        }
        let cost = block_cost(left, right, lx, rx, y, r);
        if cost < best {
            second = best;
            best = cost;
            best_d = d;
        } else if cost < second {
            second = cost;
        }
    }
    if best.is_finite() {
        Some((best_d, best, second))
    } else {
        None
    }
}

/// Dense back-projection: every validly-matched left pixel becomes a metric
/// 3D point in the left-camera frame, carrying its intensity for colouring.
pub fn dense_stereo_points(
    left: &GrayscaleImage,
    right: &GrayscaleImage,
    camera: &Camera,
    baseline: f64,
    cfg: &DenseStereoConfig,
) -> Vec<DenseStereoPoint> {
    let disparities = dense_disparity_map(left, right, cfg);
    let w = left.width();
    let h = left.height();
    let mut points = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let d = disparities[y * w + x];
            if !d.is_finite() {
                continue;
            }
            let Some(point_cam) = triangulate_stereo_pixel(
                camera,
                baseline,
                (x as f64, y as f64),
                ((x as f64) - d as f64, y as f64),
                0.0,
            ) else {
                continue;
            };
            points.push(DenseStereoPoint {
                pixel: (x, y),
                point_cam,
                disparity: d,
                intensity: at(left, x, y),
            });
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured grayscale image: a deterministic pseudo-random pattern so
    /// block matching has unambiguous structure to lock onto.
    fn textured(w: usize, h: usize) -> Vec<f32> {
        let mut px = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // cheap hash → [0,1)
                let v = ((x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663)) % 251) as f32;
                px[y * w + x] = v / 251.0;
            }
        }
        px
    }

    /// Build a right image as the left shifted so a left pixel at column x is
    /// matched by the right pixel at x-d: right(j) = left(j + d).
    fn shift_right(left_px: &[f32], w: usize, h: usize, d: usize) -> Vec<f32> {
        let mut r = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let src = x + d;
                r[y * w + x] = if src < w { left_px[y * w + src] } else { 0.0 };
            }
        }
        r
    }

    #[test]
    fn recovers_constant_disparity() {
        let (w, h, d) = (96, 64, 12);
        let lpx = textured(w, h);
        let rpx = shift_right(&lpx, w, h, d);
        let left = GrayscaleImage::new(w, h, lpx).unwrap();
        let right = GrayscaleImage::new(w, h, rpx).unwrap();
        let cfg = DenseStereoConfig {
            min_disparity: 1,
            max_disparity: 32,
            block_radius: 3,
            max_block_diff: 0.02,
            uniqueness_ratio: 1.0, // pseudo-random texture can tie; don't gate on uniqueness here
            lr_consistency_px: Some(1.5),
        };
        let disp = dense_disparity_map(&left, &right, &cfg);

        // Over the interior (away from the right-edge occlusion band where
        // right(j)=left(j+d) runs off the image), the recovered disparity must
        // be d for the vast majority of matched pixels.
        let mut matched = 0usize;
        let mut correct = 0usize;
        for y in 6..(h - 6) {
            for x in 16..(w - d - 6) {
                let v = disp[y * w + x];
                if v.is_finite() {
                    matched += 1;
                    if (v - d as f32).abs() < 1e-3 {
                        correct += 1;
                    }
                }
            }
        }
        assert!(matched > 100, "expected many matched pixels, got {matched}");
        let frac = correct as f32 / matched as f32;
        assert!(
            frac > 0.95,
            "recovered disparity wrong: {correct}/{matched} = {frac:.3} at d={d}"
        );
    }

    #[test]
    fn back_projects_to_metric_points() {
        let (w, h, d) = (96, 64, 12);
        let lpx = textured(w, h);
        let rpx = shift_right(&lpx, w, h, d);
        let left = GrayscaleImage::new(w, h, lpx).unwrap();
        let right = GrayscaleImage::new(w, h, rpx).unwrap();
        let camera = Camera::pinhole(0, w as u32, h as u32, 200.0, 200.0, 48.0, 32.0);
        let baseline = 0.1;
        let cfg = DenseStereoConfig {
            max_disparity: 32,
            max_block_diff: 0.02,
            uniqueness_ratio: 1.0,
            ..DenseStereoConfig::default()
        };
        let pts = dense_stereo_points(&left, &right, &camera, baseline, &cfg);
        assert!(!pts.is_empty());
        // Z = fx*b/d = 200*0.1/12 ≈ 1.667 m for every matched pixel.
        let z_expected = 200.0 * baseline / d as f64;
        for p in &pts {
            assert!(p.point_cam.z > 0.0);
            assert!(
                (p.point_cam.z - z_expected).abs() < 1e-6,
                "depth {} != {}",
                p.point_cam.z,
                z_expected
            );
        }
    }
}
