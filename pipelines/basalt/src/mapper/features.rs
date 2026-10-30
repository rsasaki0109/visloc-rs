//! Basalt's offline mapper image-feature frontend (M8c).
//!
//! This module follows the pinned upstream `keypoints.cpp`, `hash_bow.h`, and
//! `nfr_mapper.cpp` order at commit `0f3b2b52c807f70ff4e2973ce253c73329eea7bc`.
//! It intentionally has no dependency on the generic visual descriptor/PnP
//! stack.  The raw-u16 input is converted to 8-bit only for the
//! `goodFeaturesToTrack`-equivalent detector; orientation and descriptors read
//! the original u16 samples exactly as Basalt does.

use crate::fma::FusedMulAdd;
use std::{
    collections::BTreeMap,
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::vio::landmarks::libstdcxx_unordered_keys;
use nalgebra::{
    linalg::{Schur, SVD},
    Complex, DMatrix, DVector, Matrix3, Point2, SMatrix, Vector3, Vector4,
};
use thiserror::Error;
use visloc_core::geometry::SE3;

use crate::{camera::DoubleSphereCamera, pyramid::RawU16Image};

use super::{DescriptorMatch, OfflineMapperConfig};

/// The fixed 31x31 binary descriptor pattern used by Basalt.
const HALF_PATCH_SIZE: i32 = 15;
const EDGE_THRESHOLD: f64 = 19.0;
const STEREO_ESSENTIAL_THRESHOLD: f64 = 1e-3;

/// A normalized HashBoW entry.  Upstream stores this as
/// `std::pair<FeatureHash,double>`; `hash` is the low 16/32-bit value in the
/// `FeatureHash` bitset and `weight` is L1-normalized per image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BowEntry {
    pub hash: u32,
    pub weight: f64,
}

/// Stable image identity used by the mapper's HashBoW database query.
///
/// This is deliberately additive: the existing synthetic `Keypoint` API and
/// mapper-off path continue to use their original IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapperImageId {
    pub frame_id: u64,
    pub cam_id: u8,
}

/// Canonical semantic result of `HashBow::querry_database`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BowQueryCandidate {
    pub image: MapperImageId,
    pub score: f64,
}

/// Raw temporal match stage before/after the upstream RANSAC gate.
///
/// The exact upstream inlier model is an OpenGV randomized five-point RANSAC
/// and remains a separately documented port boundary.  This value nevertheless
/// exposes the deterministic mutual-Hamming stage and the strict min-match
/// decision used to decide whether RANSAC executes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalMatchStage {
    pub raw_matches: Vec<DescriptorMatch>,
    pub ransac_attempted: bool,
    pub raw_match_gate_passed: bool,
}

/// Result of the upstream central-relative OpenGV RANSAC stage.
///
/// The production entry point derives its seed from the current wall-clock
/// second plus a monotonic clock component, matching OpenGV's
/// `time(0)+clock()` semantics.  [`match_temporal_ransac_seeded`] is the
/// deterministic test/fixture entry point; it does not alter the production
/// default.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalRansacResult {
    pub seed: u32,
    pub ransac_model_rotation: [[f64; 3]; 3],
    pub ransac_model_translation: [f64; 3],
    pub refined_model_rotation: [[f64; 3]; 3],
    pub refined_model_translation: [f64; 3],
    /// Match-list indices selected by the final OpenGV distance test, in its
    /// pre-refinement RANSAC distance test, in its original correspondence
    /// order.  The public helper also exposes the corresponding feature-ID
    /// pairs below for fixture assertions.
    pub ransac_inlier_indices: Vec<usize>,
    pub ransac_inlier_ids: Vec<(u64, u64)>,
    pub refined_inlier_indices: Vec<usize>,
    pub refined_inlier_ids: Vec<(u64, u64)>,
    pub ransac_iterations: usize,
    pub model_found: bool,
    pub accepted: bool,
}

/// All frontend products for one mapper image.  Feature index is the stable
/// ID used by descriptors and match pairs; no ID is synthesized from frame or
/// track arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct MapperImageFeatures {
    pub corners: Vec<Point2<f64>>,
    pub corner_angles: Vec<f64>,
    pub descriptors: Vec<[u8; 32]>,
    /// Double Sphere unprojection as upstream `Vec4`: normalized xyz and a
    /// zero homogeneous fourth component.
    pub rays: Vec<[f64; 4]>,
    pub hashes: Vec<u32>,
    /// Canonicalized by hash for deterministic Rust output.  Upstream's
    /// `HashBow::compute_bow` emits an unordered-map iteration order; the
    /// semantic entries and this ordering gap are documented in M8c's report.
    pub bow_vector: Vec<BowEntry>,
}

impl MapperImageFeatures {
    pub fn len(&self) -> usize {
        self.corners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corners.is_empty()
    }
}

/// Canonical stereo output from `NfrMapper::match_stereo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StereoFeatureMatch {
    pub raw_matches: Vec<DescriptorMatch>,
    pub essential_inliers: Vec<(usize, usize)>,
    /// Upstream inserts the pair into `feature_matches` only when this count
    /// is strictly greater than 16.
    pub mapper_feature_matches_stored: bool,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum FeaturePipelineError {
    #[error("image is too small for mapper feature detection")]
    ImageTooSmall,
    #[error("feature array lengths disagree: corners={corners}, angles={angles}, descriptors={descriptors}, rays={rays}, hashes={hashes}")]
    LengthMismatch {
        corners: usize,
        angles: usize,
        descriptors: usize,
        rays: usize,
        hashes: usize,
    },
    #[error("corner {index} is outside the descriptor image border")]
    CornerOutOfBounds { index: usize },
    #[error("camera unprojection failed for corner {index}")]
    UnprojectionFailed { index: usize },
    #[error("stereo images must contain rays for every descriptor")]
    MissingRays,
    #[error("stereo transform has zero translation")]
    ZeroStereoBaseline,
}

/// Upstream's quality level: corners weaker than this fraction of the
/// image's strongest are dropped (OpenCV's `qualityLevel`).
pub const MAPPING_QUALITY_LEVEL: f32 = 0.01;

/// Detects Shi--Tomasi corners in the same role as OpenCV's
/// `goodFeaturesToTrack(image, ..., 800, 0.01, 8)` used by
/// `detectKeypointsMapping`.
///
/// The implementation keeps the pinned 8-bit conversion, 3x3 Sobel/box
/// structure, 3x3 non-maximum suppression, quality threshold, minimum-distance
/// suppression, and response-descending selection.  Coordinates are integer
/// pixel centers represented as `f64`, matching the unrefined OpenCV output.
pub fn detect_keypoints_mapping(image: &RawU16Image, num_features: usize) -> Vec<Point2<f64>> {
    detect_keypoints_mapping_with(image, num_features, MAPPING_QUALITY_LEVEL)
}

/// [`detect_keypoints_mapping`] with another quality level: lower keeps
/// weak texture that a few strong corners (lamps, windows) would otherwise
/// put below the bar.
pub fn detect_keypoints_mapping_with(
    image: &RawU16Image,
    num_features: usize,
    quality_level: f32,
) -> Vec<Point2<f64>> {
    if num_features == 0 || image.width() < 7 || image.height() < 7 {
        return Vec::new();
    }

    let width = image.width();
    let height = image.height();
    let mut response = vec![0.0_f32; width * height];

    // OpenCV's cornerMinEigenVal(CV_8U, blockSize=3, aperture=3) uses
    // BORDER_REFLECT_101, float derivatives, and scale = 1/(4*3*255).
    // Keeping the intermediate arithmetic f32 is important: the final four
    // selected points are close in response and f64/direct-convolution
    // arithmetic changes the OpenCV ordering at the 800-point cutoff.
    // Each pass runs along whole rows, reflecting only at their ends; the
    // sums keep their order, so the response is the same to the bit.
    let gray = image
        .pixels()
        .iter()
        .map(|&pixel| f32::from(pixel >> 8))
        .collect::<Vec<_>>();
    let (left, right) = (1, width - 2); // reflect101(-1) and reflect101(width)
    let scale = 1.0_f32 / (4.0 * 3.0 * 255.0);
    let mut dx_horizontal = vec![0.0_f32; width * height];
    let mut dy_horizontal = vec![0.0_f32; width * height];
    for ((row, dx_row), dy_row) in gray
        .chunks_exact(width)
        .zip(dx_horizontal.chunks_exact_mut(width))
        .zip(dy_horizontal.chunks_exact_mut(width))
    {
        // Separable Sobel kernels: d/dx = [1,2,1]ᵀ[-1,0,1],
        // d/dy = [-1,0,1]ᵀ[1,2,1].
        let mut sobel = |x: usize, before: usize, after: usize| {
            dx_row[x] = -row[before] + row[after];
            dy_row[x] = row[before] + 2.0 * row[x] + row[after];
        };
        sobel(0, left, 1);
        for x in 1..width - 1 {
            sobel(x, x - 1, x + 1);
        }
        sobel(width - 1, width - 2, right);
    }

    // The gradient products of a row, summed along it at once.
    let mut box_xx = vec![0.0_f32; width * height];
    let mut box_xy = vec![0.0_f32; width * height];
    let mut box_yy = vec![0.0_f32; width * height];
    let mut products = [
        vec![0.0_f32; width],
        vec![0.0_f32; width],
        vec![0.0_f32; width],
    ];
    for y in 0..height {
        let (above, below) = (
            reflect101(y as i32 - 1, height as i32) as usize,
            reflect101(y as i32 + 1, height as i32) as usize,
        );
        let row = |row: usize| row * width..(row + 1) * width;
        let (dx_above, dx_here, dx_below) = (
            &dx_horizontal[row(above)],
            &dx_horizontal[row(y)],
            &dx_horizontal[row(below)],
        );
        let (dy_above, dy_below) = (&dy_horizontal[row(above)], &dy_horizontal[row(below)]);
        let [xx, xy, yy] = &mut products;
        for x in 0..width {
            let dx = (dx_above[x] + 2.0 * dx_here[x] + dx_below[x]) * scale;
            let dy = (-dy_above[x] + dy_below[x]) * scale;
            xx[x] = dx * dx;
            xy[x] = dx * dy;
            yy[x] = dy * dy;
        }
        for (values, sums) in products.iter().zip([&mut box_xx, &mut box_xy, &mut box_yy]) {
            let sum = &mut sums[row(y)];
            sum[0] = values[left] + values[0] + values[1];
            for x in 1..width - 1 {
                sum[x] = values[x - 1] + values[x] + values[x + 1];
            }
            sum[width - 1] = values[width - 2] + values[width - 1] + values[right];
        }
    }
    // OpenCV computes cornerMinEigenVal for the complete image, including
    // the border.  The subsequent goodFeaturesToTrack scan starts at one
    // pixel in, but its 3x3 dilation reads those border responses; leaving
    // them at zero creates false maxima at x/y == 1.
    for (y, out) in response.chunks_exact_mut(width).enumerate() {
        let rows = [
            reflect101(y as i32 - 1, height as i32) as usize,
            y,
            reflect101(y as i32 + 1, height as i32) as usize,
        ]
        .map(|row| row * width..(row + 1) * width);
        let [xx0, xx1, xx2] = rows.clone().map(|range| &box_xx[range]);
        let [xy0, xy1, xy2] = rows.clone().map(|range| &box_xy[range]);
        let [yy0, yy1, yy2] = rows.map(|range| &box_yy[range]);
        for x in 0..width {
            let a = (xx0[x] + xx1[x] + xx2[x]) * 0.5;
            let b = xy0[x] + xy1[x] + xy2[x];
            let c = (yy0[x] + yy1[x] + yy2[x]) * 0.5;
            out[x] = (a + c) - ((a - c) * (a - c) + b * b).sqrt();
        }
    }

    let max_response = response.iter().copied().fold(0.0_f32, f32::max);
    if !max_response.is_finite() || max_response <= 0.0 {
        return Vec::new();
    }
    let quality = max_response * quality_level;
    let mut candidates = Vec::<(f32, usize, usize)>::new();
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let value = response[y * width + x];
            if value < quality {
                continue;
            }
            let mut local_max = value;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    local_max = local_max.max(response[yy * width + xx]);
                }
            }
            if value >= local_max {
                candidates.push((value, x, y));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            // OpenCV's greaterThanPtr() breaks equal-value ties by the
            // address of the float in the row-major response image, i.e.
            // later y/x pixels sort first.
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.1.cmp(&a.1))
    });

    // OpenCV's `goodFeaturesToTrack` checks candidates in response order and
    // accepts a point iff every chosen point is at least minDistance away.
    // Chosen points are kept in cells of that size, so only the 3x3 cells
    // around a candidate can hold one too close.
    const MIN_DISTANCE: usize = 8;
    let min_distance_sq = (MIN_DISTANCE * MIN_DISTANCE) as f32;
    let (cells_x, cells_y) = (width / MIN_DISTANCE + 1, height / MIN_DISTANCE + 1);
    let mut cells = vec![Vec::<(f32, f32)>::new(); cells_x * cells_y];
    let mut points = Vec::with_capacity(num_features.min(candidates.len()));
    for (_, x, y) in candidates {
        let (cell_x, cell_y) = (x / MIN_DISTANCE, y / MIN_DISTANCE);
        let far = (cell_y.saturating_sub(1)..=(cell_y + 1).min(cells_y - 1)).all(|row| {
            (cell_x.saturating_sub(1)..=(cell_x + 1).min(cells_x - 1)).all(|column| {
                cells[row * cells_x + column].iter().all(|&(px, py)| {
                    let dx = px - x as f32;
                    let dy = py - y as f32;
                    dx * dx + dy * dy >= min_distance_sq
                })
            })
        });
        if far {
            cells[cell_y * cells_x + cell_x].push((x as f32, y as f32));
            points.push(Point2::new(x as f64, y as f64));
            if points.len() == num_features {
                break;
            }
        }
    }

    // `detectKeypointsMapping` filters the already-selected OpenCV points
    // after goodFeaturesToTrack returns.  Filtering before the selection
    // would refill the 800-point budget and changes both count and order.
    points
        .into_iter()
        .filter(|point| {
            point.x >= EDGE_THRESHOLD
                && point.x < width as f64 - EDGE_THRESHOLD - 1.0
                && point.y >= EDGE_THRESHOLD
                && point.y < height as f64 - EDGE_THRESHOLD - 1.0
        })
        .collect()
}

const fn reflect101(index: i32, size: i32) -> i32 {
    if size <= 1 {
        return 0;
    }
    let period = 2 * size - 2;
    let mut value = index % period;
    if value < 0 {
        value += period;
    }
    if value >= size {
        period - value
    } else {
        value
    }
}

/// Computes the upstream intensity-centroid orientation.  `rotate_features`
/// is intentionally explicit because mapper detection passes `true`.
pub fn compute_angles(
    image: &RawU16Image,
    corners: &[Point2<f64>],
    rotate_features: bool,
) -> Vec<f64> {
    corners
        .iter()
        .map(|corner| {
            if !rotate_features {
                return 0.0;
            }
            let cx = corner.x as i32;
            let cy = corner.y as i32;
            let mut m01 = 0.0;
            let mut m10 = 0.0;
            for x in -HALF_PATCH_SIZE..=HALF_PATCH_SIZE {
                for y in -HALF_PATCH_SIZE..=HALF_PATCH_SIZE {
                    if x * x + y * y <= HALF_PATCH_SIZE * HALF_PATCH_SIZE {
                        if let Some(value) = image.pixel((cx + x) as usize, (cy + y) as usize) {
                            m01 += f64::from(y) * f64::from(value);
                            m10 += f64::from(x) * f64::from(value);
                        }
                    }
                }
            }
            m01.atan2(m10)
        })
        .collect()
}

/// Computes Basalt's 256 binary comparisons with the rotated 31x31 pattern.
pub fn compute_descriptors(
    image: &RawU16Image,
    corners: &[Point2<f64>],
    angles: &[f64],
) -> Result<Vec<[u8; 32]>, FeaturePipelineError> {
    if corners.len() != angles.len() {
        return Err(FeaturePipelineError::LengthMismatch {
            corners: corners.len(),
            angles: angles.len(),
            descriptors: 0,
            rays: 0,
            hashes: 0,
        });
    }
    let pattern = descriptor_pattern();
    let mut descriptors = Vec::with_capacity(corners.len());
    for (index, (corner, &angle)) in corners.iter().zip(angles).enumerate() {
        let cx = corner.x as i32;
        let cy = corner.y as i32;
        if cx < 19 || cy < 19 || cx >= image.width() as i32 - 19 || cy >= image.height() as i32 - 19
        {
            return Err(FeaturePipelineError::CornerOutOfBounds { index });
        }
        let (sin, cos) = angle.sin_cos();
        let mut descriptor = [0_u8; 32];
        for bit in 0..256 {
            let ax = f64::from(pattern.xa[bit]);
            let ay = f64::from(pattern.ya[bit]);
            let bx = f64::from(pattern.xb[bit]);
            let by = f64::from(pattern.yb[bit]);
            let a = (
                (cos * ax - sin * ay).round() as i32,
                (sin * ax + cos * ay).round() as i32,
            );
            let b = (
                (cos * bx - sin * by).round() as i32,
                (sin * bx + cos * by).round() as i32,
            );
            let va = image
                .pixel((cx + a.0) as usize, (cy + a.1) as usize)
                .unwrap_or(0);
            let vb = image
                .pixel((cx + b.0) as usize, (cy + b.1) as usize)
                .unwrap_or(0);
            if va < vb {
                descriptor[bit / 8] |= 1_u8 << (bit % 8);
            }
        }
        descriptors.push(descriptor);
    }
    Ok(descriptors)
}

/// Unprojects every detector corner with the configured Double Sphere model.
pub fn unproject_rays(
    camera: &DoubleSphereCamera,
    corners: &[Point2<f64>],
) -> Result<Vec<[f64; 4]>, FeaturePipelineError> {
    corners
        .iter()
        .enumerate()
        .map(|(index, corner)| {
            let ray = camera
                .unproject(corner)
                .ok_or(FeaturePipelineError::UnprojectionFailed { index })?;
            Ok([ray.x, ray.y, ray.z, 0.0])
        })
        .collect()
}

/// Exact 32-bit HashBow bit permutation filtered to 256-bit descriptors.
const HASH_PERMUTATION: [usize; 32] = [
    170, 215, 41, 38, 96, 172, 52, 1, 182, 89, 234, 98, 217, 73, 195, 113, 161, 247, 24, 75, 7,
    232, 49, 196, 144, 69, 3, 86, 94, 201, 107, 251,
];

pub fn hash_descriptor(descriptor: &[u8; 32], bits: u8) -> u32 {
    let bits = usize::from(bits.min(32));
    let mut hash = 0_u32;
    for index in 0..bits {
        let source = HASH_PERMUTATION[index];
        if descriptor[source / 8] & (1_u8 << (source % 8)) != 0 {
            hash |= 1_u32 << index;
        }
    }
    hash
}

/// Computes per-descriptor hashes and L1-normalized HashBoW entries.
pub fn compute_hash_bow(descriptors: &[[u8; 32]], bits: u8) -> (Vec<u32>, Vec<BowEntry>) {
    let hashes = descriptors
        .iter()
        .map(|descriptor| hash_descriptor(descriptor, bits))
        .collect::<Vec<_>>();
    let mut counts = BTreeMap::<u32, usize>::new();
    for &hash in &hashes {
        *counts.entry(hash).or_default() += 1;
    }
    let denominator = descriptors.len() as f64;
    let bow_vector = if denominator == 0.0 {
        Vec::new()
    } else {
        counts
            .into_iter()
            .map(|(hash, count)| BowEntry {
                hash,
                weight: count as f64 / denominator,
            })
            .collect()
    };
    (hashes, bow_vector)
}

/// Reproduces the semantic score calculation of
/// `HashBow::querry_database`.  The upstream inverted index is unordered;
/// this implementation returns the candidate set in canonical score/ID order
/// so callers can compare it without depending on bucket traversal order.
pub fn query_bow_candidates<'a>(
    query_id: MapperImageId,
    query: &MapperImageFeatures,
    database: &[(MapperImageId, &'a MapperImageFeatures)],
    num_results: usize,
) -> Vec<BowQueryCandidate> {
    let mut candidates = Vec::new();
    for &(image, features) in database {
        if image.frame_id >= query_id.frame_id {
            continue;
        }
        let mut score = 0.0;
        let mut shared = false;
        for query_entry in &query.bow_vector {
            if let Some(database_entry) = features
                .bow_vector
                .iter()
                .find(|entry| entry.hash == query_entry.hash)
            {
                shared = true;
                score += (query_entry.weight - database_entry.weight).abs()
                    - query_entry.weight.abs()
                    - database_entry.weight.abs();
            }
        }
        if shared {
            candidates.push(BowQueryCandidate {
                image,
                score: -score / 2.0,
            });
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.image.cmp(&right.image))
    });
    candidates.truncate(num_results);
    candidates.sort_by(|left, right| {
        left.image
            .cmp(&right.image)
            .then_with(|| right.score.total_cmp(&left.score))
    });
    candidates
}

/// Runs the deterministic part of upstream `NfrMapper::match_all` for one
/// query/candidate pair.  Upstream executes OpenGV RANSAC only when the strict
/// `raw_matches > mapper_min_matches` gate passes; the returned stage records
/// that exact decision while leaving model-dependent inliers to the explicit
/// OpenGV parity boundary.
pub fn match_temporal_stage(
    left: &MapperImageFeatures,
    right: &MapperImageFeatures,
    config: OfflineMapperConfig,
) -> TemporalMatchStage {
    let mut raw_matches = mutual_descriptor_matches(
        left.descriptors.as_slice(),
        right.descriptors.as_slice(),
        config,
    );
    raw_matches.sort_by_key(|match_| (match_.left, match_.right));
    let raw_match_gate_passed = raw_matches.len() > config.min_matches;
    TemporalMatchStage {
        raw_matches,
        ransac_attempted: raw_match_gate_passed,
        raw_match_gate_passed,
    }
}

const OPENGV_RANSAC_MAX_ITERATIONS: usize = 100;
const OPENGV_RANSAC_PROBABILITY: f64 = 0.99;
const OPENGV_RANSAC_SAMPLE_SIZE: usize = 8;
const OPENGV_RANSAC_MINIMAL_SIZE: usize = 5;

type Matrix9x4 = SMatrix<f64, 9, 4>;
type Matrix10 = SMatrix<f64, 10, 10>;
type Matrix10x20 = SMatrix<f64, 10, 20>;
type ComplexMatrix10 = SMatrix<Complex<f64>, 10, 10>;
type ComplexVector10 = SMatrix<Complex<f64>, 10, 1>;

#[derive(Debug, Clone, Copy)]
struct RelativePoseModel {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

// This is the monomial order used by OpenGV's generated five-point
// `composeA`: cubic monomials first, followed by quadratic, linear, and
// constant terms after substituting x3=1.
const STEWENIUS_MONOMIALS: [(u8, u8, u8); 20] = [
    (3, 0, 0),
    (2, 1, 0),
    (1, 2, 0),
    (0, 3, 0),
    (2, 0, 1),
    (1, 1, 1),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (0, 0, 3),
    (2, 0, 0),
    (1, 1, 0),
    (0, 2, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0, 0, 2),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
];

const LINEAR_EXPONENTS: [(u8, u8, u8); 4] = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)];

fn stewenius_monomial_index(exponents: (u8, u8, u8)) -> Option<usize> {
    STEWENIUS_MONOMIALS
        .iter()
        .position(|candidate| *candidate == exponents)
}

fn stewenius_product3(a: &[f64; 4], b: &[f64; 4], c: &[f64; 4]) -> [f64; 20] {
    let mut result = [0.0_f64; 20];
    for ia in 0..4 {
        for ib in 0..4 {
            for ic in 0..4 {
                let exponents = (
                    LINEAR_EXPONENTS[ia].0 + LINEAR_EXPONENTS[ib].0 + LINEAR_EXPONENTS[ic].0,
                    LINEAR_EXPONENTS[ia].1 + LINEAR_EXPONENTS[ib].1 + LINEAR_EXPONENTS[ic].1,
                    LINEAR_EXPONENTS[ia].2 + LINEAR_EXPONENTS[ib].2 + LINEAR_EXPONENTS[ic].2,
                );
                let index = stewenius_monomial_index(exponents)
                    .expect("five-point product has a cubic monomial");
                result[index] += a[ia] * b[ib] * c[ic];
            }
        }
    }
    result
}

fn stewenius_accumulate(target: &mut [f64; 20], scale: f64, term: &[f64; 20]) {
    for (dst, src) in target.iter_mut().zip(term) {
        *dst += scale * src;
    }
}

fn stewenius_compose_a(ee: &Matrix9x4) -> Matrix10x20 {
    let mut entries = [[[0.0_f64; 4]; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            entries[row][col] = [
                ee[(row * 3 + col, 0)],
                ee[(row * 3 + col, 1)],
                ee[(row * 3 + col, 2)],
                ee[(row * 3 + col, 3)],
            ];
        }
    }

    let mut result = Matrix10x20::zeros();
    // The first nine OpenGV rows are the essential-matrix cubic constraint
    // `E E^T E - 1/2 trace(E E^T) E = 0`.
    for row in 0..3 {
        for col in 0..3 {
            // OpenGV's generated composeA stores the nine cubic constraints
            // in Eigen's column-major coefficient order.
            let equation_row = col * 3 + row;
            let mut polynomial = [0.0_f64; 20];
            for i in 0..3 {
                for j in 0..3 {
                    let product =
                        stewenius_product3(&entries[row][i], &entries[j][i], &entries[j][col]);
                    stewenius_accumulate(&mut polynomial, 1.0, &product);
                    let trace_product =
                        stewenius_product3(&entries[i][j], &entries[i][j], &entries[row][col]);
                    stewenius_accumulate(&mut polynomial, -0.5, &trace_product);
                }
            }
            for monomial in 0..20 {
                result[(equation_row, monomial)] = polynomial[monomial];
            }
        }
    }

    // The tenth OpenGV row is det(E)=0.
    let determinant_terms = [
        (1.0, (0, 0, 1, 1, 2, 2)),
        (1.0, (0, 1, 1, 2, 2, 0)),
        (1.0, (0, 2, 1, 0, 2, 1)),
        (-1.0, (0, 2, 1, 1, 2, 0)),
        (-1.0, (0, 1, 1, 0, 2, 2)),
        (-1.0, (0, 0, 1, 2, 2, 1)),
    ];
    let mut determinant = [0.0_f64; 20];
    for (scale, (r0, c0, r1, c1, r2, c2)) in determinant_terms {
        let product = stewenius_product3(&entries[r0][c0], &entries[r1][c1], &entries[r2][c2]);
        stewenius_accumulate(&mut determinant, scale, &product);
    }
    for monomial in 0..20 {
        result[(9, monomial)] = determinant[monomial];
    }
    result
}

fn stewenius_nullspace(
    left_rays: &[[f64; 4]],
    right_rays: &[[f64; 4]],
    matches: &[DescriptorMatch],
    sample: &[usize],
) -> Option<Matrix9x4> {
    if sample.len() < OPENGV_RANSAC_MINIMAL_SIZE {
        return None;
    }
    let mut q = DMatrix::<f64>::zeros(OPENGV_RANSAC_MINIMAL_SIZE, 9);
    for (row, &match_index) in sample.iter().take(OPENGV_RANSAC_MINIMAL_SIZE).enumerate() {
        let correspondence = matches.get(match_index)?;
        let f1 = left_rays.get(correspondence.left as usize)?;
        let f2 = right_rays.get(correspondence.right as usize)?;
        // OpenGV's fivept_stewenius deliberately inverts the input adapter:
        // f=f2, fprime=f1.  Its row is row-major f[c]*fprime[r].
        for r in 0..3 {
            for c in 0..3 {
                q[(row, r * 3 + c)] = f2[c] * f1[r];
            }
        }
    }

    stewenius_eigen_full_v_last4_f64(&q)
}

fn stewenius_eigen_col_piv_qr_f64(
    q: &DMatrix<f64>,
) -> (DMatrix<f64>, Vec<f64>, Vec<usize>, DMatrix<f64>) {
    let size = q.nrows();
    let scale = q.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let mut qr = q.transpose().map(|value| value / scale);
    let mut h = vec![0.0_f64; size];
    let mut cols = vec![0_usize; size];
    let mut norms = (0..size)
        .map(|column| qr.column(column).norm())
        .collect::<Vec<_>>();
    let mut direct_norms = norms.clone();
    let norm_downdate_threshold = f64::EPSILON.sqrt();
    for k in 0..size {
        let mut pivot = k;
        for column in k + 1..size {
            if norms[column] > norms[pivot] {
                pivot = column;
            }
        }
        cols[k] = pivot;
        if pivot != k {
            qr.swap_columns(k, pivot);
            norms.swap(k, pivot);
            direct_norms.swap(k, pivot);
        }
        let c0 = qr[(k, k)];
        let tail_sq = (k + 1..9)
            .map(|row| qr[(row, k)] * qr[(row, k)])
            .sum::<f64>();
        if tail_sq <= f64::MIN_POSITIVE {
            h[k] = 0.0;
            for row in k + 1..9 {
                qr[(row, k)] = 0.0;
            }
        } else {
            let mut beta = (c0 * c0 + tail_sq).sqrt();
            if c0 >= 0.0 {
                beta = -beta;
            }
            for row in k + 1..9 {
                qr[(row, k)] /= c0 - beta;
            }
            h[k] = (beta - c0) / beta;
            qr[(k, k)] = beta;
        }
        for column in k + 1..size {
            let mut dot = qr[(k, column)];
            for row in k + 1..9 {
                dot += qr[(row, k)] * qr[(row, column)];
            }
            let update = h[k] * dot;
            qr[(k, column)] -= update;
            for row in k + 1..9 {
                qr[(row, column)] -= qr[(row, k)] * update;
            }
        }
        for column in k + 1..size {
            if norms[column] != 0.0 {
                let ratio = qr[(k, column)].abs() / norms[column];
                let temp = ((1.0 + ratio) * (1.0 - ratio)).max(0.0);
                let temp2 = temp * (norms[column] / direct_norms[column]).powi(2);
                if temp2 <= norm_downdate_threshold {
                    direct_norms[column] = (k + 1..9)
                        .map(|row| qr[(row, column)] * qr[(row, column)])
                        .sum::<f64>()
                        .sqrt();
                    norms[column] = direct_norms[column];
                } else {
                    norms[column] *= temp.sqrt();
                }
            }
        }
    }
    let mut matrix_q = DMatrix::<f64>::identity(9, 9);
    for k in (0..size).rev() {
        for column in k..9 {
            let mut dot = matrix_q[(k, column)];
            for row in k + 1..9 {
                dot += qr[(row, k)] * matrix_q[(row, column)];
            }
            let update = h[k] * dot;
            matrix_q[(k, column)] -= update;
            for row in k + 1..9 {
                matrix_q[(row, column)] -= qr[(row, k)] * update;
            }
        }
    }
    let mut permutation = (0..size).collect::<Vec<_>>();
    for k in 0..size {
        permutation.swap(k, cols[k]);
    }
    (qr, h, permutation, matrix_q)
}

fn stewenius_eigen_full_v_last4_f64(q: &DMatrix<f64>) -> Option<Matrix9x4> {
    if q.nrows() != OPENGV_RANSAC_MINIMAL_SIZE
        || q.ncols() != 9
        || q.iter().any(|value| !value.is_finite())
        || q.iter().all(|value| *value == 0.0)
    {
        return None;
    }
    // Eigen's wide dynamic JacobiSVD scales Q before its default
    // ColPivHouseholderQR preconditioner. The four trailing columns of the
    // resulting full Q are already the exact nullspace columns requested by
    // V.rightCols(4); the later 5x5 Jacobi sweeps never touch them.
    let (_, _, _, matrix_q) = stewenius_eigen_col_piv_qr_f64(q);
    let mut result = Matrix9x4::zeros();
    for row in 0..9 {
        for column in 0..4 {
            result[(row, column)] = matrix_q[(row, column + 5)];
        }
    }
    Some(result)
}

// A RANSAC hypothesis must not make the entire mapper wait indefinitely:
// `Schur::new` iterates without limit and can spin forever on non-finite or
// non-convergent input.  This generous per-decomposition budget is a safety
// ceiling, not a convergence tolerance change.
const STEWENIUS_SCHUR_MAX_ITERATIONS: usize = 10_000;

fn stewenius_schur_bounded(
    matrix: ComplexMatrix10,
    max_iterations: usize,
) -> Option<(ComplexMatrix10, ComplexMatrix10)> {
    assert!(
        max_iterations > 0,
        "zero would enable unbounded Schur iteration"
    );
    if !matrix.iter().all(|x| x.re.is_finite() && x.im.is_finite()) {
        return None;
    }
    // Same epsilon as Schur::new; only non-convergent hypotheses are dropped.
    let (q, triangular) = Schur::try_new(matrix, f64::EPSILON, max_iterations)?.unpack();
    q.iter()
        .chain(triangular.iter())
        .all(|x| x.re.is_finite() && x.im.is_finite())
        .then_some((q, triangular))
}

fn stewenius_eigenvectors(matrix: ComplexMatrix10) -> Vec<ComplexVector10> {
    // The complex Schur form is upper triangular.  Back-substitution gives
    // the right eigenvector for each diagonal eigenvalue, then Q maps it back
    // to the original matrix.  This is the same EigenSolver result consumed
    // by OpenGV's fivept_stewenius_main, without the nalgebra Eigen helper's
    // unconditional diagnostic println.
    let Some((q, triangular)) = stewenius_schur_bounded(matrix, STEWENIUS_SCHUR_MAX_ITERATIONS)
    else {
        return Vec::new();
    };
    let mut vectors = Vec::with_capacity(10);
    for column in 0..10 {
        let eigenvalue = triangular[(column, column)];
        let mut vector = ComplexVector10::zeros();
        vector[column] = Complex::new(1.0, 0.0);
        for row in (0..column).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for k in row + 1..=column {
                sum += triangular[(row, k)] * vector[k];
            }
            let denominator = triangular[(row, row)] - eigenvalue;
            vector[row] = if denominator.norm() > 1e-12 {
                -sum / denominator
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        let mapped = q * vector;
        let norm = mapped.norm();
        vectors.push(if norm > 0.0 {
            mapped / Complex::new(norm, 0.0)
        } else {
            mapped
        });
    }
    vectors
}

fn stewenius_essential_solutions(
    left_rays: &[[f64; 4]],
    right_rays: &[[f64; 4]],
    matches: &[DescriptorMatch],
    sample: &[usize],
) -> Option<Vec<Matrix3<f64>>> {
    let ee = stewenius_nullspace(left_rays, right_rays, matches, sample)?;
    let polynomial = stewenius_compose_a(&ee);
    let a1 = polynomial.fixed_view::<10, 10>(0, 0).into_owned();
    let a2 = polynomial.fixed_view::<10, 10>(0, 10).into_owned();
    let a3 = a1.full_piv_lu().solve(&a2)?;

    let mut m = Matrix10::zeros();
    for row in 0..3 {
        for col in 0..10 {
            m[(row, col)] = -a3[(row, col)];
        }
    }
    for row in 0..2 {
        for col in 0..10 {
            m[(row + 3, col)] = -a3[(row + 4, col)];
        }
    }
    for col in 0..10 {
        m[(5, col)] = -a3[(7, col)];
    }
    m[(6, 0)] = 1.0;
    m[(7, 1)] = 1.0;
    m[(8, 3)] = 1.0;
    m[(9, 6)] = 1.0;

    let complex_matrix = m.map(|value| Complex::new(value, 0.0));
    let eigenvectors = stewenius_eigenvectors(complex_matrix);
    let mut solutions = Vec::with_capacity(eigenvectors.len());
    for vector in eigenvectors {
        let denominator = vector[9];
        if denominator.norm() <= 1e-14 {
            continue;
        }
        let solution = [
            vector[6] / denominator,
            vector[7] / denominator,
            vector[8] / denominator,
            Complex::new(1.0, 0.0),
        ];
        let mut evec = [Complex::new(0.0, 0.0); 9];
        for row in 0..9 {
            for column in 0..4 {
                evec[row] += Complex::new(ee[(row, column)], 0.0) * solution[column];
            }
        }
        let squared_norm = evec
            .iter()
            .fold(Complex::new(0.0, 0.0), |sum, value| sum + *value * *value);
        let norm = squared_norm.sqrt();
        if norm.norm() <= 1e-14 {
            continue;
        }
        let mut essential = Matrix3::zeros();
        for row in 0..3 {
            for column in 0..3 {
                essential[(row, column)] = (evec[row * 3 + column] / norm).re;
            }
        }
        solutions.push(essential);
    }
    if solutions.is_empty() {
        None
    } else {
        Some(solutions)
    }
}

fn triangulate2(
    left: Vector3<f64>,
    right: Vector3<f64>,
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
) -> Option<Vector3<f64>> {
    let right_unrotated = opengv_eigen_matvec3(rotation, right);
    let a00 = opengv_eigen_dot3(&left, &left);
    let a10 = opengv_eigen_dot3(&left, &right_unrotated);
    let a01 = -a10;
    let a11 = -opengv_eigen_dot3(&right_unrotated, &right_unrotated);
    let b0 = opengv_eigen_dot3(&translation, &left);
    let b1 = opengv_eigen_dot3(&translation, &right_unrotated);
    // Eigen's optimized fixed-size determinant contracts the first product
    // with the negated second product.  Keep that FMA placement: the
    // reciprocal is measurably different for the near-degenerate rows even
    // when the rounded determinant itself prints as the same f64.
    let determinant = a00.fused_mul_add(a11, -(a10 * a01));
    if determinant.abs() <= f64::EPSILON {
        return None;
    }
    // Keep the restored production expression order for the LM numerical
    // difference path; the pinned Eigen packet codegen is recorded separately
    // by the M8c assembly probe.
    let invdet = 1.0 / determinant;
    let inverse00 = a11 * invdet;
    let inverse01 = -a01 * invdet;
    let inverse10 = -a10 * invdet;
    let inverse11 = a00 * invdet;
    let lambda0 = inverse01.fused_mul_add(b1, inverse00 * b0);
    let lambda1 = inverse11.fused_mul_add(b1, inverse10 * b0);
    // Keep Eigen's two temporaries and grouping: OpenGV computes `xm` first,
    // `xn = t12 + lambda[1] * f2_unrotated` second, then `(xm + xn) / 2`.
    // Associating this as `(xm + t12) + ...` changes the last bits of every
    // residual and can move the ill-conditioned LM refinement onto another
    // trust-region path.
    let xm = lambda0 * left;
    let xn = Vector3::new(
        right_unrotated[0].fused_mul_add(lambda1, translation[0]),
        right_unrotated[1].fused_mul_add(lambda1, translation[1]),
        right_unrotated[2].fused_mul_add(lambda1, translation[2]),
    );
    Some((xm + xn) / 2.0)
}

/// Reproduce Eigen's fixed-size AVX double packet path for a three-vector
/// dot. Eigen multiplies the first two lanes as a packet, adds those lanes,
/// then contracts the third lane with that partial sum. nalgebra's scalar
/// reduction is a different rounding path (`p0 + p1 + p2`).
#[inline(always)]
fn opengv_eigen_dot3(left: &Vector3<f64>, right: &Vector3<f64>) -> f64 {
    let first_two = left[0] * right[0] + left[1] * right[1];
    left[2].fused_mul_add(right[2], first_two)
}

/// Eigen's fixed-size `Vector3d::norm()` squares the first two lanes as a
/// packet, reduces them, then uses one scalar FMA for lane three.
#[inline(always)]
fn opengv_eigen_norm3(value: &Vector3<f64>) -> f64 {
    let first_two = value[0] * value[0] + value[1] * value[1];
    value[2].fused_mul_add(value[2], first_two).sqrt()
}

/// Fixed 3x3-by-3x1 Eigen product with the same AVX row packet and scalar
/// tail ordering as the pinned OpenGV build.  Rows zero/one are accumulated
/// by columns; the scalar tail is the same two-lane packet reduction followed
/// by the third-column FMA (`m22*x2 + (m21*x1 + m20*x0)`).
#[inline(always)]
fn opengv_eigen_matvec3(matrix: Matrix3<f64>, value: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(
        matrix[(0, 2)].fused_mul_add(
            value[2],
            matrix[(0, 1)].fused_mul_add(value[1], matrix[(0, 0)] * value[0]),
        ),
        matrix[(1, 2)].fused_mul_add(
            value[2],
            matrix[(1, 1)].fused_mul_add(value[1], matrix[(1, 0)] * value[0]),
        ),
        matrix[(2, 2)].fused_mul_add(
            value[2],
            matrix[(2, 1)].fused_mul_add(value[1], matrix[(2, 0)] * value[0]),
        ),
    )
}

/// Fixed 3x4-by-4x1 Eigen product used by OpenGV's homogeneous inverse
/// transform.  The first two rows are packetized across columns; the scalar
/// third row reduces two pairs before adding them.
#[inline(always)]
fn opengv_eigen_mat34_vec4(matrix: SMatrix<f64, 3, 4>, value: Vector4<f64>) -> Vector3<f64> {
    Vector3::new(
        matrix[(0, 3)].fused_mul_add(
            value[3],
            matrix[(0, 2)].fused_mul_add(
                value[2],
                matrix[(0, 1)].fused_mul_add(value[1], matrix[(0, 0)] * value[0]),
            ),
        ),
        matrix[(1, 3)].fused_mul_add(
            value[3],
            matrix[(1, 2)].fused_mul_add(
                value[2],
                matrix[(1, 1)].fused_mul_add(value[1], matrix[(1, 0)] * value[0]),
            ),
        ),
        matrix[(2, 2)].fused_mul_add(value[2], matrix[(2, 3)] * value[3])
            + matrix[(2, 0)].fused_mul_add(value[0], matrix[(2, 1)] * value[1]),
    )
}

fn relative_pose_error(left: Vector3<f64>, right: Vector3<f64>, model: RelativePoseModel) -> f64 {
    let Some(point) = triangulate2(left, right, model.rotation, model.translation) else {
        return f64::INFINITY;
    };
    let first_norm = opengv_eigen_norm3(&point);
    // OpenGV materializes the inverse 4x4 transform first and then applies
    // it to the homogeneous point.  Keep the translation multiply/add
    // separated in that order rather than factoring it as Rᵀ*(point-t).
    let inverse_rotation = model.rotation.transpose();
    let inverse_translation = -opengv_eigen_matvec3(inverse_rotation, model.translation);
    let mut inverse_solution = SMatrix::<f64, 3, 4>::zeros();
    inverse_solution
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&inverse_rotation);
    inverse_solution
        .column_mut(3)
        .copy_from(&inverse_translation);
    let point_hom = Vector4::new(point[0], point[1], point[2], 1.0);
    let second = opengv_eigen_mat34_vec4(inverse_solution, point_hom);
    let second_norm = opengv_eigen_norm3(&second);
    if first_norm <= f64::EPSILON || second_norm <= f64::EPSILON {
        return f64::INFINITY;
    }
    let first_reprojection = point / first_norm;
    let second_reprojection = second / second_norm;
    (1.0 - opengv_eigen_dot3(&left, &first_reprojection))
        + (1.0 - opengv_eigen_dot3(&right, &second_reprojection))
}

fn relative_pose_inliers(
    left_rays: &[[f64; 4]],
    right_rays: &[[f64; 4]],
    matches: &[DescriptorMatch],
    model: RelativePoseModel,
    threshold: f64,
) -> Vec<usize> {
    matches
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            let left = left_rays.get(correspondence.left as usize)?;
            let right = right_rays.get(correspondence.right as usize)?;
            let error = relative_pose_error(
                Vector3::new(left[0], left[1], left[2]),
                Vector3::new(right[0], right[1], right[2]),
                model,
            );
            (error < threshold).then_some(index)
        })
        .collect()
}

fn stewenius_model_from_sample(
    left_rays: &[[f64; 4]],
    right_rays: &[[f64; 4]],
    matches: &[DescriptorMatch],
    sample: &[usize],
) -> Option<RelativePoseModel> {
    let essentials = stewenius_essential_solutions(left_rays, right_rays, matches, sample)?;
    let mut best_quality = f64::INFINITY;
    let mut best_model = None;
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    for essential in essentials {
        // SVD::new also iterates without limit; non-finite input can hang it.
        if !essential.iter().all(|value| value.is_finite()) {
            continue;
        }
        let svd = SVD::new(essential, true, true);
        let singular_values = svd.singular_values;
        let Some(u) = svd.u else { continue };
        let Some(v_t) = svd.v_t else { continue };
        let v = v_t.transpose();
        let scale = singular_values[0];
        let mut ra = u * w * v.transpose();
        let mut rb = u * w.transpose() * v.transpose();
        if ra.determinant() < 0.0 {
            ra = -ra;
        }
        if rb.determinant() < 0.0 {
            rb = -rb;
        }
        let ta = scale * u.column(2).into_owned();
        let tb = -ta;
        let candidates = [
            RelativePoseModel {
                rotation: ra,
                translation: ta,
            },
            RelativePoseModel {
                rotation: rb,
                translation: ta,
            },
            RelativePoseModel {
                rotation: ra,
                translation: tb,
            },
            RelativePoseModel {
                rotation: rb,
                translation: tb,
            },
        ];
        for candidate in candidates {
            let mut quality = 0.0;
            let mut valid = true;
            for &match_index in sample.iter().take(OPENGV_RANSAC_SAMPLE_SIZE) {
                let Some(correspondence) = matches.get(match_index) else {
                    valid = false;
                    break;
                };
                let Some(left_ray) = left_rays.get(correspondence.left as usize) else {
                    valid = false;
                    break;
                };
                let Some(right_ray) = right_rays.get(correspondence.right as usize) else {
                    valid = false;
                    break;
                };
                let Some(point) = triangulate2(
                    Vector3::new(left_ray[0], left_ray[1], left_ray[2]),
                    Vector3::new(right_ray[0], right_ray[1], right_ray[2]),
                    candidate.rotation,
                    candidate.translation,
                ) else {
                    valid = false;
                    break;
                };
                let point_norm = opengv_eigen_norm3(&point);
                let inverse_rotation = candidate.rotation.transpose();
                let inverse_translation =
                    -opengv_eigen_matvec3(inverse_rotation, candidate.translation);
                let second = opengv_eigen_matvec3(inverse_rotation, point) + inverse_translation;
                let second_norm = opengv_eigen_norm3(&second);
                if point_norm <= f64::EPSILON || second_norm <= f64::EPSILON {
                    valid = false;
                    break;
                }
                quality +=
                    2.0 - opengv_eigen_dot3(
                        &Vector3::new(left_ray[0], left_ray[1], left_ray[2]),
                        &(point / point_norm),
                    ) - opengv_eigen_dot3(
                        &Vector3::new(right_ray[0], right_ray[1], right_ray[2]),
                        &(second / second_norm),
                    );
            }
            if valid && quality < best_quality {
                best_quality = quality;
                best_model = Some(candidate);
            }
        }
    }
    best_model
}

/// The libstdc++ `std::mt19937` used by OpenGV, including its exact seeding
/// and tempering constants.  OpenGV then asks a libstdc++
/// `uniform_int_distribution<int>(0, INT_MAX)` for each draw.  GCC 11 takes
/// libstdc++'s Lemire `_S_nd` exact-32-bit path here: the range is `2^31`,
/// the low-word threshold is zero, and every generator value maps to its
/// high half (`next_u32() >> 1`) with no rejection, including both extrema.
#[derive(Debug, Clone)]
struct OpenGvMt19937 {
    state: [u32; 624],
    index: usize,
}

impl OpenGvMt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0_u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253_u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let value = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            self.state[i] = self.state[(i + 397) % 624] ^ (value >> 1);
            if value & 1 != 0 {
                self.state[i] ^= 0x9908_b0df;
            }
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn uniform_int_max(&mut self) -> u32 {
        self.next_u32() >> 1
    }
}

fn opengv_time_seed() -> u32 {
    // OpenGV uses `static_cast<unsigned>(time(0)) +
    // static_cast<unsigned>(clock())`.  Rust has no portable process-CPU
    // clock in std, so the nanosecond component supplies the same
    // per-invocation variability while preserving the wall-clock seed
    // semantics for the production API.  Golden tests use the explicit seed
    // API below and never depend on this value.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() as u32).wrapping_add(now.subsec_nanos())
}

fn opengv_draw_sample(
    shuffled_indices: &mut [usize],
    sample_size: usize,
    rng: &mut OpenGvMt19937,
) -> Vec<usize> {
    for i in 0..sample_size {
        let offset = (rng.uniform_int_max() as usize) % (shuffled_indices.len() - i);
        shuffled_indices.swap(i, i + offset);
    }
    shuffled_indices[..sample_size].to_vec()
}

fn opengv_cayley2rot(cayley: Vector3<f64>) -> Matrix3<f64> {
    // The pinned OpenGV binary emits these expressions as scalar FMAs.  Keep
    // the same operation order: the LM's numerical-difference probes are
    // sensitive to a few ulps in this conversion even when the unperturbed
    // residual is unchanged.
    let c0 = cayley[0];
    let c1 = cayley[1];
    let c2 = cayley[2];
    let one_plus_c0sq = c0.fused_mul_add(c0, 1.0);
    let one_minus_c0sq = (-c0).fused_mul_add(c0, 1.0);
    let scale = c2.fused_mul_add(c2, c1.fused_mul_add(c1, one_plus_c0sq));
    let inv_scale = 1.0 / scale;
    let r00 = (-c2).fused_mul_add(c2, (-c1).fused_mul_add(c1, one_plus_c0sq));
    let r11 = (-c2).fused_mul_add(c2, c1.fused_mul_add(c1, one_minus_c0sq));
    let r22 = c2.fused_mul_add(c2, (-c1).fused_mul_add(c1, one_minus_c0sq));
    let r01 = c0.fused_mul_add(c1, -c2) * 2.0;
    let r02 = c0.fused_mul_add(c2, c1) * 2.0;
    let r10 = c0.fused_mul_add(c1, c2) * 2.0;
    let r12 = c1.fused_mul_add(c2, -c0) * 2.0;
    let r20 = c0.fused_mul_add(c2, -c1) * 2.0;
    let r21 = c1.fused_mul_add(c2, c0) * 2.0;
    Matrix3::new(
        inv_scale * r00,
        inv_scale * r01,
        inv_scale * r02,
        inv_scale * r10,
        inv_scale * r11,
        inv_scale * r12,
        inv_scale * r20,
        inv_scale * r21,
        inv_scale * r22,
    )
}

fn opengv_eigen_inverse3(matrix: Matrix3<f64>) -> Matrix3<f64> {
    let cofactor = |row: usize, column: usize| {
        let i1 = (row + 1) % 3;
        let i2 = (row + 2) % 3;
        let j1 = (column + 1) % 3;
        let j2 = (column + 2) % 3;
        matrix[(i1, j1)] * matrix[(i2, j2)] - matrix[(i1, j2)] * matrix[(i2, j1)]
    };
    let cofactors_col0 = Vector3::new(cofactor(0, 0), cofactor(1, 0), cofactor(2, 0));
    let determinant = cofactors_col0[0] * matrix[(0, 0)]
        + cofactors_col0[1] * matrix[(1, 0)]
        + cofactors_col0[2] * matrix[(2, 0)];
    let invdet = 1.0 / determinant;
    Matrix3::new(
        cofactors_col0[0] * invdet,
        cofactor(1, 0) * invdet,
        cofactor(2, 0) * invdet,
        cofactor(0, 1) * invdet,
        cofactor(1, 1) * invdet,
        cofactor(2, 1) * invdet,
        cofactor(0, 2) * invdet,
        cofactor(1, 2) * invdet,
        cofactor(2, 2) * invdet,
    )
}

fn opengv_rot2cayley(rotation: Matrix3<f64>) -> Vector3<f64> {
    let identity = Matrix3::identity();
    let c = (rotation - identity) * opengv_eigen_inverse3(rotation + identity);
    Vector3::new(-c[(1, 2)], c[(0, 2)], -c[(0, 1)])
}

fn opengv_lm_blue_norm(vector: &DVector<f64>) -> f64 {
    // Eigen's `blueNorm()` is the classic Blue's algorithm, not the
    // max-coefficient scaling used by `stableNorm()`.  For the ordinary
    // double-valued residual/Jacobian ranges here this reduces to a direct
    // amed sum, but retaining the same bucket thresholds and reduction order
    // keeps the LM trust-region scalar sequence aligned with Eigen.
    let n = vector.len().max(1) as f64;
    let b1 = 2.0_f64.powi(-511);
    let b2 = 2.0_f64.powi(486) / n;
    let s1m = 2.0_f64.powi(511);
    let s2m = 2.0_f64.powi(-538);
    let relerr = 2.0_f64.powi(-26);
    let mut asml = 0.0;
    let mut amed = 0.0;
    let mut abig = 0.0;
    for &value in vector.iter() {
        let ax = value.abs();
        if ax > b2 {
            let scaled = ax * s2m;
            abig += scaled * scaled;
        } else if ax < b1 {
            let scaled = ax * s1m;
            asml += scaled * scaled;
        } else {
            amed += ax * ax;
        }
    }
    if amed.is_nan() {
        return amed;
    }
    if abig > 0.0 {
        abig = abig.sqrt();
        if abig > f64::MAX {
            return abig;
        }
        if amed > 0.0 {
            abig /= s2m;
            amed = amed.sqrt();
        } else {
            return abig / s2m;
        }
    } else if asml > 0.0 {
        if amed > 0.0 {
            abig = amed.sqrt();
            amed = asml.sqrt() / s1m;
        } else {
            return asml.sqrt() / s1m;
        }
    } else {
        return amed.sqrt();
    }
    asml = abig.min(amed);
    abig = abig.max(amed);
    if asml <= abig * relerr {
        abig
    } else {
        abig * (1.0 + (asml / abig).powi(2)).sqrt()
    }
}

/// Eigen's LM uses `stableNorm()` for residual and scaled-state norms while
/// reserving `blueNorm()` for the QR/LM work vectors.  They are numerically
/// very close for this problem, but preserving the two reductions matters
/// once the optimizer runs to its 1000-function-evaluation limit.
fn opengv_lm_stable_norm(vector: &DVector<f64>) -> f64 {
    let scale = vector.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        0.0
    } else {
        scale
            * vector
                .iter()
                .map(|value| {
                    let scaled = value / scale;
                    scaled * scaled
                })
                .sum::<f64>()
                .sqrt()
    }
}

fn opengv_lm_backsolve_upper(matrix: &DMatrix<f64>, values: &mut DVector<f64>) {
    for row in (0..6).rev() {
        let mut value = values[row];
        for column in row + 1..6 {
            value -= matrix[(row, column)] * values[column];
        }
        values[row] = value / matrix[(row, row)];
    }
}

fn opengv_lm_forwardsolve_transpose(matrix: &DMatrix<f64>, values: &mut DVector<f64>) {
    for row in 0..6 {
        let mut value = values[row];
        for column in 0..row {
            value -= matrix[(column, row)] * values[column];
        }
        values[row] = value / matrix[(row, row)];
    }
}

/// Solves the trust-region parameter subproblem used by Eigen's
/// `internal::lmpar2`.  OpenGV's refinement uses Eigen's QR/LM implementation,
/// rather than a normal-equation Gauss--Newton step; keeping the QR solve here
/// is material for the seeded cam1 hypotheses.
fn opengv_lmpar2(
    r: &DMatrix<f64>,
    pivots: &[usize; 6],
    diagonal: &DVector<f64>,
    qtf: &DVector<f64>,
    delta: f64,
    parameter: &mut f64,
) -> DVector<f64> {
    const N: usize = 6;
    let mut work = qtf.clone();
    opengv_lm_backsolve_upper(r, &mut work);
    let mut x = DVector::<f64>::zeros(N);
    for column in 0..N {
        x[pivots[column]] = work[column];
    }

    let mut scaled = diagonal.component_mul(&x);
    let mut step_norm = opengv_lm_blue_norm(&scaled);
    let mut fp = step_norm - delta;
    if fp <= 0.1 * delta {
        *parameter = 0.0;
        return x;
    }

    let mut lower = 0.0;
    if step_norm > 0.0 {
        let mut work = DVector::<f64>::zeros(N);
        for column in 0..N {
            work[column] = diagonal[pivots[column]] * scaled[pivots[column]] / step_norm;
        }
        opengv_lm_forwardsolve_transpose(r, &mut work);
        let norm = opengv_lm_blue_norm(&work);
        if norm > 0.0 {
            lower = fp / delta / (norm * norm);
        }
    }

    let mut work = DVector::<f64>::zeros(N);
    for column in 0..N {
        let mut value = 0.0;
        for row in 0..=column {
            value += r[(row, column)] * qtf[row];
        }
        work[column] = value / diagonal[pivots[column]];
    }
    // Eigen's LMpar upper-bound gradient uses stableNorm(), while the
    // trust-region dxnorm/correction norms use blueNorm().
    let gradient_norm = opengv_lm_stable_norm(&work);
    let tiny = f64::MIN_POSITIVE;
    let mut upper = if gradient_norm > 0.0 {
        gradient_norm / delta
    } else {
        tiny / delta.min(0.1)
    };
    *parameter = parameter.max(lower).min(upper);
    if *parameter == 0.0 {
        *parameter = gradient_norm / step_norm;
    }

    let mut iteration = 0;
    let mut s = r.clone();
    loop {
        iteration += 1;
        if *parameter == 0.0 {
            *parameter = (tiny).max(0.001 * upper);
        }
        let damping = diagonal.map(|value| parameter.sqrt() * value);
        let mut sdiag = DVector::<f64>::zeros(N);
        opengv_qrsolv(&mut s, pivots, &damping, qtf, &mut x, &mut sdiag);

        scaled = diagonal.component_mul(&x);
        step_norm = opengv_lm_blue_norm(&scaled);
        let previous_fp = fp;
        fp = step_norm - delta;
        if fp.abs() <= 0.1 * delta
            || (lower == 0.0 && fp <= previous_fp && previous_fp < 0.0)
            || iteration == 10
        {
            break;
        }

        let mut correction = DVector::<f64>::zeros(N);
        for column in 0..N {
            correction[column] = diagonal[pivots[column]] * scaled[pivots[column]] / step_norm;
        }
        for row in 0..N {
            correction[row] /= sdiag[row];
            let value = correction[row];
            for lower_row in row + 1..N {
                correction[lower_row] -= s[(lower_row, row)] * value;
            }
        }
        let correction_norm = opengv_lm_blue_norm(&correction);
        if correction_norm == 0.0 {
            break;
        }
        let correction_parameter = fp / delta / (correction_norm * correction_norm);
        if fp > 0.0 {
            lower = lower.max(*parameter);
        } else if fp < 0.0 {
            upper = upper.min(*parameter);
        }
        *parameter = lower.max(*parameter + correction_parameter);
    }
    x
}

fn opengv_qrsolv(
    s: &mut DMatrix<f64>,
    pivots: &[usize; 6],
    diagonal: &DVector<f64>,
    qtb: &DVector<f64>,
    x: &mut DVector<f64>,
    sdiag: &mut DVector<f64>,
) {
    const N: usize = 6;
    let mut work = qtb.clone();
    let mut original_diagonal = DVector::<f64>::zeros(N);
    for row in 0..N {
        original_diagonal[row] = s[(row, row)];
    }
    // Eigen's lmqrsolv mirrors the upper-triangular R into the lower
    // triangle on every call.  The lower triangle contains the previous
    // damping solve's Givens transforms, so doing this only once in lmpar2
    // changes subsequent trust-region corrections.
    for row in 0..N {
        for column in 0..row {
            s[(row, column)] = s[(column, row)];
        }
    }
    for column in 0..N {
        let pivot = pivots[column];
        if diagonal[pivot] == 0.0 {
            break;
        }
        for row in column..N {
            sdiag[row] = 0.0;
        }
        sdiag[column] = diagonal[pivot];
        let mut qtb_pivot = 0.0;
        for row in column..N {
            let original = s[(row, row)];
            let p = -original;
            let q = sdiag[row];
            let (c, sn) = opengv_make_givens(p, q);
            s[(row, row)] = c * original + sn * q;
            let temp = c * work[row] + sn * qtb_pivot;
            qtb_pivot = -sn * work[row] + c * qtb_pivot;
            work[row] = temp;
            for lower_row in row + 1..N {
                let temp = c * s[(lower_row, row)] + sn * sdiag[lower_row];
                sdiag[lower_row] = -sn * s[(lower_row, row)] + c * sdiag[lower_row];
                s[(lower_row, row)] = temp;
            }
        }
    }

    let mut nonsingular = 0;
    while nonsingular < N && sdiag[nonsingular] != 0.0 {
        nonsingular += 1;
    }
    for row in nonsingular..N {
        work[row] = 0.0;
    }
    for row in (0..nonsingular).rev() {
        let mut value = work[row];
        for column in row + 1..nonsingular {
            value -= s[(column, row)] * work[column];
        }
        work[row] = value / s[(row, row)];
    }
    for column in 0..N {
        x[pivots[column]] = work[column];
    }
    for row in 0..N {
        sdiag[row] = s[(row, row)];
        s[(row, row)] = original_diagonal[row];
    }
}

fn opengv_make_givens(p: f64, q: f64) -> (f64, f64) {
    if q == 0.0 {
        (if p < 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else if p == 0.0 {
        (0.0, if q < 0.0 { 1.0 } else { -1.0 })
    } else if p.abs() > q.abs() {
        let t = q / p;
        let mut u = (1.0 + t * t).sqrt();
        if p < 0.0 {
            u = -u;
        }
        let c = 1.0 / u;
        (c, -t * c)
    } else {
        let t = p / q;
        let mut u = (1.0 + t * t).sqrt();
        if q < 0.0 {
            u = -u;
        }
        let sn = -1.0 / u;
        (-t * sn, sn)
    }
}

/// Eigen's `ColPivHouseholderQR` pivot policy differs from nalgebra's
/// `ColPivQR`: Eigen selects the largest updated *column norm*, while
/// nalgebra selects the largest individual trailing coefficient.  The
/// distinction is observable in OpenGV's six-parameter LM Jacobians, so keep
/// this compact Eigen-compatible factorization local to the optimizer.
struct OpengvEigenQr {
    qr: DMatrix<f64>,
    pivots: [usize; 6],
    householder: [f64; 6],
}

fn opengv_eigen_cpqr(jacobian: &DMatrix<f64>) -> OpengvEigenQr {
    let (rows, cols) = jacobian.shape();
    debug_assert_eq!(cols, 6);
    let mut qr = jacobian.clone();
    let mut pivots = [0_usize; 6];
    let mut householder = [0.0_f64; 6];
    let mut column_ids = (0..cols).collect::<Vec<_>>();
    let mut norms_updated = vec![0.0_f64; cols];
    let mut norms_direct = vec![0.0_f64; cols];
    for column in 0..cols {
        let norm = (0..rows)
            .map(|row| qr[(row, column)] * qr[(row, column)])
            .sum::<f64>()
            .sqrt();
        norms_updated[column] = norm;
        norms_direct[column] = norm;
    }

    let norm_downdate_threshold = f64::EPSILON.sqrt();
    for k in 0..6 {
        let mut biggest = k;
        let mut biggest_norm = norms_updated[k];
        for column in k + 1..cols {
            if norms_updated[column] > biggest_norm {
                biggest = column;
                biggest_norm = norms_updated[column];
            }
        }
        if biggest != k {
            qr.swap_columns(k, biggest);
            norms_updated.swap(k, biggest);
            norms_direct.swap(k, biggest);
            column_ids.swap(k, biggest);
        }
        pivots[k] = column_ids[k];

        let c0 = qr[(k, k)];
        let tail_sq_norm = (k + 1..rows)
            .map(|row| qr[(row, k)] * qr[(row, k)])
            .sum::<f64>();
        let tol = f64::MIN_POSITIVE;
        let (tau, beta) = if tail_sq_norm <= tol && c0 * c0 <= tol {
            (0.0, c0)
        } else {
            let mut beta = (c0 * c0 + tail_sq_norm).sqrt();
            if c0 >= 0.0 {
                beta = -beta;
            }
            let denominator = c0 - beta;
            for row in k + 1..rows {
                qr[(row, k)] /= denominator;
            }
            ((beta - c0) / beta, beta)
        };
        householder[k] = tau;
        qr[(k, k)] = beta;

        if tau != 0.0 && k + 1 < cols {
            // Eigen's applyHouseholderOnTheLeft computes each workspace
            // element from the old row/bottom values, then updates the row
            // and bottom in separate passes.
            for column in k + 1..cols {
                let mut work = qr[(k, column)];
                for row in k + 1..rows {
                    work += qr[(row, k)] * qr[(row, column)];
                }
                qr[(k, column)] -= tau * work;
                for row in k + 1..rows {
                    qr[(row, column)] -= tau * qr[(row, k)] * work;
                }
            }
        }

        for column in k + 1..cols {
            if norms_updated[column] != 0.0 {
                let temp = qr[(k, column)].abs() / norms_updated[column];
                let temp = (1.0 + temp) * (1.0 - temp);
                let temp = temp.max(0.0);
                let temp2 = temp
                    * (norms_updated[column] / norms_direct[column])
                    * (norms_updated[column] / norms_direct[column]);
                if temp2 <= norm_downdate_threshold {
                    let norm = (k + 1..rows)
                        .map(|row| qr[(row, column)] * qr[(row, column)])
                        .sum::<f64>()
                        .sqrt();
                    norms_updated[column] = norm;
                    norms_direct[column] = norm;
                } else {
                    norms_updated[column] *= temp.sqrt();
                }
            }
        }
    }
    OpengvEigenQr {
        qr,
        pivots,
        householder,
    }
}

fn opengv_eigen_qt_mul(qr: &OpengvEigenQr, values: &DVector<f64>) -> DVector<f64> {
    let mut result = values.clone();
    let rows = qr.qr.nrows();
    for k in 0..6 {
        let tau = qr.householder[k];
        if tau == 0.0 {
            continue;
        }
        let mut work = result[k];
        for row in k + 1..rows {
            work += qr.qr[(row, k)] * result[row];
        }
        result[k] -= tau * work;
        for row in k + 1..rows {
            result[row] -= tau * qr.qr[(row, k)] * work;
        }
    }
    result
}

fn opengv_optimize_pose(
    initial: RelativePoseModel,
    left_rays: &[[f64; 4]],
    right_rays: &[[f64; 4]],
    matches: &[DescriptorMatch],
    inliers: &[usize],
) -> RelativePoseModel {
    // `opengv_eigen_cpqr` performs a column-pivoted QR of a (rows x 6)
    // Jacobian (one row per inlier residual, six columns for the
    // translation+Cayley-rotation parameters) and indexes qr[(k, k)] for
    // k in 0..6, so it requires rows >= 6. `raw_matches.len() <
    // OPENGV_RANSAC_SAMPLE_SIZE` in `run_opengv_ransac` only bounds the
    // *candidate* match count, not the RANSAC consensus set eventually
    // passed here as `inliers`: `best_count` starts below zero, so even a
    // 1-5-point consensus set is accepted as "best" and forwarded to this
    // 6-DOF refinement, previously panicking with "Matrix index out of
    // bounds" on any frame pair whose best RANSAC model has fewer than 6
    // inliers (observed on LaMAria's wider-FOV imagery, which produces
    // more marginal two-view geometries than EuRoC's data). Skipping
    // refinement below 6 inliers -- the same "return the unrefined model"
    // behavior already used for the zero-inlier case -- is the minimal
    // generalization of the existing guard.
    if inliers.len() < 6 {
        return initial;
    }
    let mut x = DVector::<f64>::zeros(6);
    x[0] = initial.translation[0];
    x[1] = initial.translation[1];
    x[2] = initial.translation[2];
    let cayley = opengv_rot2cayley(initial.rotation);
    x[3] = cayley[0];
    x[4] = cayley[1];
    x[5] = cayley[2];
    let residuals = |parameters: &DVector<f64>| -> DVector<f64> {
        let model = RelativePoseModel {
            rotation: opengv_cayley2rot(Vector3::new(parameters[3], parameters[4], parameters[5])),
            translation: Vector3::new(parameters[0], parameters[1], parameters[2]),
        };
        DVector::from_iterator(
            inliers.len(),
            inliers.iter().map(|&index| {
                let correspondence = matches[index];
                let left = left_rays[correspondence.left as usize];
                let right = right_rays[correspondence.right as usize];
                relative_pose_error(
                    Vector3::new(left[0], left[1], left[2]),
                    Vector3::new(right[0], right[1], right[2]),
                    model,
                )
            }),
        )
    };
    let mut fvec = residuals(&x);
    let mut function_evaluations = 1_usize;
    let mut function_norm = opengv_lm_stable_norm(&fvec);
    let mut parameter = 0.0_f64;
    let mut iteration = 1_usize;
    let mut diagonal = DVector::<f64>::zeros(6);
    let mut delta = 0.0_f64;
    let tolerance = 10.0 * f64::EPSILON;

    while function_evaluations < 1000 {
        let rows = fvec.len();
        let mut jacobian = DMatrix::<f64>::zeros(rows, 6);
        for column in 0..6 {
            let mut perturbed = x.clone();
            let mut step = f64::EPSILON.sqrt() * x[column].abs();
            // Eigen NumericalDiff substitutes sqrt(epsilon) only when the
            // computed h is exactly zero; it does not clamp nonzero h.
            if step == 0.0 {
                step = f64::EPSILON.sqrt();
            }
            perturbed[column] += step;
            let forward = residuals(&perturbed);
            function_evaluations += 1;
            for row in 0..rows {
                jacobian[(row, column)] = (forward[row] - fvec[row]) / step;
            }
        }
        // Eigen::NumericalDiff::df() evaluates the functor at the current
        // point once before its six forward perturbations.  `fvec` is already
        // that value here, so no residual recomputation is needed, but the
        // evaluation budget must still include it: Eigen's LM stops at
        // maxfev=1000 based on this count.
        function_evaluations += 1;
        let column_norms = DVector::from_iterator(
            6,
            (0..6).map(|column| {
                let values =
                    DVector::from_iterator(rows, (0..rows).map(|row| jacobian[(row, column)]));
                opengv_lm_blue_norm(&values)
            }),
        );
        let qr = opengv_eigen_cpqr(&jacobian);
        let r = DMatrix::from_fn(6, 6, |row, column| qr.qr[(row, column)]);
        let pivots = qr.pivots;
        // Eigen applies `householderQ().adjoint()` directly to the residual
        // vector.  Use nalgebra's in-place reflector path instead of forming
        // Q and multiplying it, preserving the same sequence of scalar
        // updates for qᵀf.
        let qtf_full = opengv_eigen_qt_mul(&qr, &fvec);
        let qtf = qtf_full.rows(0, 6).into_owned();
        if iteration == 1 {
            diagonal = column_norms.map(|value| if value == 0.0 { 1.0 } else { value });
            let xnorm = opengv_lm_blue_norm(&diagonal.component_mul(&x));
            delta = 100.0 * xnorm;
            if delta == 0.0 {
                delta = 100.0;
            }
        }
        let mut gradient_norm = 0.0_f64;
        if function_norm != 0.0 {
            for column in 0..6 {
                if column_norms[pivots[column]] != 0.0 {
                    let dot = (0..=column)
                        .map(|row| r[(row, column)] * (qtf[row] / function_norm))
                        .sum::<f64>();
                    gradient_norm = gradient_norm.max((dot / column_norms[pivots[column]]).abs());
                }
            }
        }
        if gradient_norm <= 0.0 {
            break;
        }
        for column in 0..6 {
            diagonal[column] = diagonal[column].max(column_norms[column]);
        }

        loop {
            let mut lm_parameter = parameter;
            let mut step = opengv_lmpar2(&r, &pivots, &diagonal, &qtf, delta, &mut lm_parameter);
            step = -step;
            // Eigen's LevenbergMarquardt shrinks the initial trust-region
            // radius to the first trial step.  This is observable here:
            // OpenGV starts with the Gauss--Newton step inside the default
            // radius, then carries that smaller radius into the next LM
            // iteration.  Omitting this update lets the Rust port take a
            // materially different second step and eventually lose the
            // boundary inlier at the 5e-5 OpenGV gate.
            if iteration == 1 {
                let first_step_norm = opengv_lm_stable_norm(&diagonal.component_mul(&step));
                delta = delta.min(first_step_norm);
            }
            let candidate = &x + &step;
            let candidate_residuals = residuals(&candidate);
            function_evaluations += 1;
            let candidate_norm = opengv_lm_stable_norm(&candidate_residuals);
            let actual = if 0.1 * candidate_norm < function_norm {
                1.0 - (candidate_norm / function_norm).powi(2)
            } else {
                -1.0
            };
            let mut projected = DVector::<f64>::zeros(6);
            // Eigen computes R * (P^{-1} p) here.  The previous port used
            // the transpose (Rᵀ * z), which leaves the trial step itself
            // plausible but corrupts the predicted reduction and therefore
            // the LM trust-region update after the first accepted step.
            for row in 0..6 {
                projected[row] = (row..6)
                    .map(|column| r[(row, column)] * step[pivots[column]])
                    .sum::<f64>();
            }
            let step_norm = opengv_lm_stable_norm(&diagonal.component_mul(&step));
            let projected_norm = opengv_lm_stable_norm(&projected);
            let temp1 = (projected_norm / function_norm).powi(2);
            let temp2 = (lm_parameter.sqrt() * step_norm / function_norm).powi(2);
            let predicted = temp1 + temp2 / 0.5;
            let directional = -(temp1 + temp2);
            let ratio = if predicted != 0.0 {
                actual / predicted
            } else {
                0.0
            };
            if ratio <= 0.25 {
                let mut factor = if actual >= 0.0 {
                    0.5
                } else {
                    0.5 * directional / (directional + 0.5 * actual)
                };
                if 0.1 * candidate_norm >= function_norm || factor < 0.1 {
                    factor = 0.1;
                }
                delta = factor * delta.min(step_norm / 0.1);
                parameter = lm_parameter / factor;
            } else if !(lm_parameter != 0.0 && ratio < 0.75) {
                delta = step_norm / 0.5;
                parameter = 0.5 * lm_parameter;
            } else {
                parameter = lm_parameter;
            }
            if ratio >= 1e-4 {
                x = candidate;
                fvec = candidate_residuals;
                function_norm = candidate_norm;
                iteration += 1;
            }
            let xnorm = opengv_lm_stable_norm(&diagonal.component_mul(&x));
            // Eigen checks the relative-reduction tests independently from
            // the trust-region-radius test.  Keeping the intermediate
            // return points matters when a trial lands on the tiny residual
            // plateau before the radius has collapsed.
            if actual.abs() <= tolerance
                && predicted <= tolerance
                && 0.5 * ratio <= 1.0
                && delta <= tolerance * xnorm
            {
                break;
            }
            if actual.abs() <= tolerance && predicted <= tolerance && 0.5 * ratio <= 1.0 {
                break;
            }
            if delta <= tolerance * xnorm {
                break;
            }
            if function_evaluations >= 1000 {
                break;
            }
            if actual.abs() <= f64::EPSILON && predicted <= f64::EPSILON && 0.5 * ratio <= 1.0 {
                break;
            }
            if delta <= f64::EPSILON * xnorm {
                break;
            }
            if gradient_norm <= f64::EPSILON {
                break;
            }
            if ratio >= 1e-4 {
                break;
            }
        }
        if function_evaluations >= 1000 {
            break;
        }
    }
    RelativePoseModel {
        rotation: opengv_cayley2rot(Vector3::new(x[3], x[4], x[5])),
        translation: Vector3::new(x[0], x[1], x[2]),
    }
}

/// Relative pose `T_left_right` (`p_left = R * p_right + t`) from matched
/// unit bearings, with the same OpenGV Stewenius 5-point RANSAC and nonlinear
/// refinement the mapper uses in [`run_opengv_ransac`]. Correspondence `i`
/// pairs `left[i]` with `right[i]`; `threshold` is OpenGV's `1 - cos` angular
/// error. Returns the refined rotation, unit translation and refined inlier
/// indices, or `None` with fewer than eight correspondences or no model.
/// Used by the visual-only two-view bootstrap (`docs/mono_visual_only_vio.md`).
pub(crate) fn relative_pose_from_bearings(
    left: &[Vector3<f64>],
    right: &[Vector3<f64>],
    threshold: f64,
    seed: u32,
) -> Option<(Matrix3<f64>, Vector3<f64>, Vec<usize>)> {
    if left.len() != right.len() || left.len() < OPENGV_RANSAC_SAMPLE_SIZE {
        return None;
    }
    let ray = |v: &Vector3<f64>| {
        let v = v.normalize();
        [v.x, v.y, v.z, 0.0]
    };
    let left_rays = left.iter().map(ray).collect::<Vec<_>>();
    let right_rays = right.iter().map(ray).collect::<Vec<_>>();
    let matches = (0..left.len() as u64)
        .map(|index| DescriptorMatch {
            left: index,
            right: index,
            distance: 0,
        })
        .collect::<Vec<_>>();
    let mut rng = OpenGvMt19937::new(seed);
    let mut shuffled = (0..matches.len()).collect::<Vec<_>>();
    let (mut iterations, mut skipped, mut k) = (0_usize, 0_usize, 1.0_f64);
    let mut best: Option<(RelativePoseModel, Vec<usize>)> = None;
    while (iterations as f64) < k && skipped < OPENGV_RANSAC_MAX_ITERATIONS * 10 {
        let sample = opengv_draw_sample(&mut shuffled, OPENGV_RANSAC_SAMPLE_SIZE, &mut rng);
        let Some(model) = stewenius_model_from_sample(&left_rays, &right_rays, &matches, &sample)
        else {
            skipped += 1;
            continue;
        };
        let inliers = relative_pose_inliers(&left_rays, &right_rays, &matches, model, threshold);
        if best.as_ref().is_none_or(|(_, b)| inliers.len() > b.len()) {
            let weight = inliers.len() as f64 / matches.len() as f64;
            let no_outliers = (1.0 - weight.powi(OPENGV_RANSAC_SAMPLE_SIZE as i32))
                .clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            k = (1.0 - OPENGV_RANSAC_PROBABILITY).ln() / no_outliers.ln();
            best = Some((model, inliers));
        }
        iterations += 1;
        if iterations > OPENGV_RANSAC_MAX_ITERATIONS {
            break;
        }
    }
    let (model, inliers) = best?;
    let refined = opengv_optimize_pose(model, &left_rays, &right_rays, &matches, &inliers);
    let inliers = relative_pose_inliers(&left_rays, &right_rays, &matches, refined, threshold);
    let norm = refined.translation.norm();
    (norm > f64::EPSILON).then(|| (refined.rotation, refined.translation / norm, inliers))
}

fn run_opengv_ransac(
    left: &MapperImageFeatures,
    right: &MapperImageFeatures,
    raw_matches: &[DescriptorMatch],
    config: OfflineMapperConfig,
    seed: u32,
) -> TemporalRansacResult {
    let mut result = TemporalRansacResult {
        seed,
        ransac_model_rotation: [[0.0; 3]; 3],
        ransac_model_translation: [0.0; 3],
        refined_model_rotation: [[0.0; 3]; 3],
        refined_model_translation: [0.0; 3],
        ransac_inlier_indices: Vec::new(),
        ransac_inlier_ids: Vec::new(),
        refined_inlier_indices: Vec::new(),
        refined_inlier_ids: Vec::new(),
        ransac_iterations: 0,
        model_found: false,
        accepted: false,
    };
    if raw_matches.len() < OPENGV_RANSAC_SAMPLE_SIZE {
        return result;
    }
    let mut rng = OpenGvMt19937::new(seed);
    let mut shuffled = (0..raw_matches.len()).collect::<Vec<_>>();
    let mut iterations = 0_usize;
    let mut skipped = 0_usize;
    let max_skip = OPENGV_RANSAC_MAX_ITERATIONS * 10;
    let mut k = 1.0_f64;
    let mut best_count = 0_i32.saturating_sub(i32::MAX);
    let mut best_model = None;
    let mut best_inliers = Vec::new();

    while (iterations as f64) < k && skipped < max_skip {
        let sample = opengv_draw_sample(&mut shuffled, OPENGV_RANSAC_SAMPLE_SIZE, &mut rng);
        let Some(model) =
            stewenius_model_from_sample(&left.rays, &right.rays, raw_matches, &sample)
        else {
            skipped += 1;
            continue;
        };
        let inliers = relative_pose_inliers(
            &left.rays,
            &right.rays,
            raw_matches,
            model,
            config.ransac_threshold,
        );
        let count = inliers.len() as i32;
        if count > best_count {
            best_count = count;
            best_model = Some(model);
            best_inliers = inliers;
            let weight = best_count as f64 / raw_matches.len() as f64;
            let no_outliers = (1.0 - weight.powi(OPENGV_RANSAC_SAMPLE_SIZE as i32))
                .clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            k = (1.0 - OPENGV_RANSAC_PROBABILITY).ln() / no_outliers.ln();
        }
        iterations += 1;
        if iterations > OPENGV_RANSAC_MAX_ITERATIONS {
            break;
        }
    }
    result.ransac_iterations = iterations;
    let Some(model) = best_model else {
        return result;
    };
    let refined = opengv_optimize_pose(model, &left.rays, &right.rays, raw_matches, &best_inliers);
    let final_inliers = relative_pose_inliers(
        &left.rays,
        &right.rays,
        raw_matches,
        refined,
        config.ransac_threshold,
    );
    result.model_found = true;
    result.accepted = final_inliers.len() >= config.min_matches;
    result.ransac_inlier_indices = best_inliers;
    result.ransac_inlier_ids = result
        .ransac_inlier_indices
        .iter()
        .map(|&index| {
            let correspondence = raw_matches[index];
            (correspondence.left, correspondence.right)
        })
        .collect();
    result.refined_inlier_indices = final_inliers;
    result.refined_inlier_ids = result
        .refined_inlier_indices
        .iter()
        .map(|&index| {
            let correspondence = raw_matches[index];
            (correspondence.left, correspondence.right)
        })
        .collect();
    let refined_translation_norm = refined.translation.norm();
    let reported_refined_translation = if refined_translation_norm > f64::EPSILON {
        refined.translation / refined_translation_norm
    } else {
        refined.translation
    };
    for row in 0..3 {
        for column in 0..3 {
            result.ransac_model_rotation[row][column] = model.rotation[(row, column)];
            result.refined_model_rotation[row][column] = refined.rotation[(row, column)];
        }
        result.ransac_model_translation[row] = model.translation[row];
        // NfrMapper::findInliersRansac stores T_i_j with normalized
        // translation after the final distance selection.  Keep the raw
        // refined scale internal to the distance gate, but expose the same
        // normalized model coefficient in the parity result.
        result.refined_model_translation[row] = reported_refined_translation[row];
    }
    result
}

/// Runs the upstream central-relative OpenGV RANSAC with an explicit seed.
/// This is the deterministic golden/test API; the default production helper
/// below derives the upstream wall-clock seed.
pub fn match_temporal_ransac_seeded(
    left: &MapperImageFeatures,
    right: &MapperImageFeatures,
    config: OfflineMapperConfig,
    seed: u32,
) -> TemporalRansacResult {
    // Keep the source `std::unordered_map` traversal order here.  The
    // explicit public stage report canonicalizes its raw vector for the M8c
    // summary fixture, but upstream feeds this order directly to OpenGV's
    // seeded RANSAC sampler.
    let raw_matches = mutual_descriptor_matches(&left.descriptors, &right.descriptors, config);
    run_opengv_ransac(left, right, &raw_matches, config, seed)
}

/// Production-compatible temporal RANSAC entry point.  OpenGV's default
/// `randomSeed=true` is `time(0)+clock()`; the Rust helper uses equivalent
/// wall-clock invocation variability.  Golden fixtures must use the seeded
/// function to remain deterministic.
pub fn match_temporal_ransac(
    left: &MapperImageFeatures,
    right: &MapperImageFeatures,
    config: OfflineMapperConfig,
) -> TemporalRansacResult {
    match_temporal_ransac_seeded(left, right, config, opengv_time_seed())
}

/// Runs the full upstream image frontend in its fixed order.
pub fn extract_mapper_features(
    image: &RawU16Image,
    camera: &DoubleSphereCamera,
    config: OfflineMapperConfig,
) -> Result<MapperImageFeatures, FeaturePipelineError> {
    extract_mapper_features_with(image, camera, config, MAPPING_QUALITY_LEVEL)
}

/// [`extract_mapper_features`] with the detector's quality level
/// ([`detect_keypoints_mapping_with`]).
pub fn extract_mapper_features_with(
    image: &RawU16Image,
    camera: &DoubleSphereCamera,
    config: OfflineMapperConfig,
    quality_level: f32,
) -> Result<MapperImageFeatures, FeaturePipelineError> {
    let corners = detect_keypoints_mapping_with(image, config.max_points, quality_level);
    let corner_angles = compute_angles(image, &corners, true);
    let descriptors = compute_descriptors(image, &corners, &corner_angles)?;
    let rays = unproject_rays(camera, &corners)?;
    let (hashes, bow_vector) = compute_hash_bow(&descriptors, config.bow_bits);
    Ok(MapperImageFeatures {
        corners,
        corner_angles,
        descriptors,
        rays,
        hashes,
        bow_vector,
    })
}

/// Computes the calibrated right-camera pose in the left-camera frame and its
/// essential matrix, matching `computeEssential(T_0_1, E)`.
pub fn compute_essential(t_0_1: &SE3) -> Result<Matrix3<f64>, FeaturePipelineError> {
    let norm = t_0_1.translation.norm();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(FeaturePipelineError::ZeroStereoBaseline);
    }
    let t = t_0_1.translation / norm;
    let hat = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    Ok(hat * t_0_1.rotation.to_rotation_matrix().into_inner())
}

/// Applies exact mutual Hamming matching and known-pose essential filtering.
pub fn match_stereo_features(
    left: &MapperImageFeatures,
    right: &MapperImageFeatures,
    t_0_1: &SE3,
    config: OfflineMapperConfig,
) -> Result<StereoFeatureMatch, FeaturePipelineError> {
    if left.descriptors.len() != left.rays.len() || right.descriptors.len() != right.rays.len() {
        return Err(FeaturePipelineError::MissingRays);
    }
    let essential = compute_essential(t_0_1)?;
    let mut raw_matches = mutual_descriptor_matches(&left.descriptors, &right.descriptors, config);
    raw_matches.sort_by_key(|match_| (match_.left, match_.right));

    let mut essential_inliers = raw_matches
        .iter()
        .filter_map(|match_| {
            let l = left.rays.get(match_.left as usize)?.as_ref();
            let r = right.rays.get(match_.right as usize)?.as_ref();
            let lv = Vector3::new(l[0], l[1], l[2]);
            let rv = Vector3::new(r[0], r[1], r[2]);
            (lv.dot(&(essential * rv)).abs() < STEREO_ESSENTIAL_THRESHOLD)
                .then_some((match_.left as usize, match_.right as usize))
        })
        .collect::<Vec<_>>();
    essential_inliers.sort_unstable();
    Ok(StereoFeatureMatch {
        mapper_feature_matches_stored: essential_inliers.len() > 16,
        raw_matches,
        essential_inliers,
    })
}

pub fn mutual_descriptor_matches(
    left: &[[u8; 32]],
    right: &[[u8; 32]],
    config: OfflineMapperConfig,
) -> Vec<DescriptorMatch> {
    fn one_way(
        source: &[[u8; 32]],
        target: &[[u8; 32]],
        threshold: u32,
        ratio: f64,
    ) -> Vec<(usize, usize)> {
        let mut matches = BTreeMap::new();
        for (source_index, source_descriptor) in source.iter().enumerate() {
            let mut best_index = usize::MAX;
            let mut best_distance = 500_u32;
            let mut second_distance = 500_u32;
            for (target_index, target_descriptor) in target.iter().enumerate() {
                let distance = hamming_distance(source_descriptor, target_descriptor);
                // This is intentionally <=, matching upstream's tie behavior
                // (the latest target wins the best slot).
                if distance <= best_distance {
                    second_distance = best_distance;
                    best_distance = distance;
                    best_index = target_index;
                } else if distance < second_distance {
                    second_distance = distance;
                }
            }
            if best_index != usize::MAX
                && best_distance < threshold
                && best_distance as f64 * ratio <= second_distance as f64
            {
                matches.insert(source_index, best_index);
            }
        }
        // `matchFastHelper` stores the accepted source indices in a fresh
        // libstdc++ `std::unordered_map<int, int>`.  Its iteration order is
        // the order consumed by the mutual-match loop and therefore by
        // OpenGV RANSAC; a BTreeMap would preserve the set but perturb the
        // seeded model/inlier boundary on the full run.
        libstdcxx_unordered_keys(matches.keys().map(|&key| key as u64))
            .into_iter()
            .map(|key| {
                let key = key as usize;
                (key, matches[&key])
            })
            .collect()
    }

    let forward = one_way(left, right, config.max_hamming, config.second_best_ratio);
    let reverse = one_way(right, left, config.max_hamming, config.second_best_ratio)
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    forward
        .into_iter()
        .filter_map(|(left_index, right_index)| {
            // Upstream uses `matches_2_1[kv.second]` (operator[]), which
            // inserts a missing reverse key with value zero.  Preserve that
            // observable corner case instead of replacing it with a strict
            // `get` lookup; feature index zero can therefore accept the same
            // fallback as the pinned C++ implementation.
            (reverse.get(&right_index).copied().unwrap_or_default() == left_index).then_some(
                DescriptorMatch {
                    left: left_index as u64,
                    right: right_index as u64,
                    distance: hamming_distance(&left[left_index], &right[right_index]),
                },
            )
        })
        .collect()
}

struct DescriptorPattern {
    xa: [i8; 256],
    ya: [i8; 256],
    xb: [i8; 256],
    yb: [i8; 256],
}

// Values are signed offsets + 13 encoded as A..AA.  Keeping the upstream
// pattern as compact literals makes accidental ID-derived descriptors
// impossible while avoiding a generated source dependency at runtime.
const PATTERN_XA: &str = "VRCUPOLAAXACUJAEZKHYRSQFLAGJDSSOWRPJFRNAKHVNUAXHXAAQSMQPAAAGTELBQGKPCMSJEBXUGJUGAKUAOPJMUOWMAUZTSPQPWFCOTPTQUCDIDVRDRLIUEIVEOULYBQSNENMSQAIJTGAORLPLRHKURAUUGFAPXHVPCBCSLMADKPEJJHTAYUMJGAGFIAOOWSMEMAVPUDDRQJSRENBQDVFPXTGKMKFRPTQYKRPDAATNAEASPMWYQMQASVUDUWUM";
const PATTERN_YA: &str = "KPWBAGDAKRFUUIPNHTAAUKGGYZQPBBHNYUMBIYFLLWZWIHUKEVNQUUDJNGQZDMISDGLWATKAHDPZAWMTYUFGKHQAOMOEAUIQABVTBRZZEQQKVIYFSMHZLNFHAAFCFJOHEUSJZUPYSJWGSTTDOLBAODASLWOFJYTRIIKBLANKAFCLWKATZCKYYIZFOBLSMUSNZFYKDOCAADFHZPAAWQOPDABPTVXEAGLPIEMMNCJHUZNMQVHEUHSKNRHNVWJRQGNH";
const PATTERN_XB: &str = "WUFZPOLCBYFEZKBGZLJZSXTHMFIKHTURYRRLGWOFLJXOYCZHZFFUXOSQABCJZGNGVJMSINSJEFZZHKZIBLZCZQLOVQZMDXZUTPRZXGJPUQYVWHIKEZTFTLIXFIWEOWMZHUXPIPOUTFKKVHIQVPZNWKMZSEVUGGBQZHWPDGDYMNBDLQJKLJTIZZNKHFHHJFSXXXOHOFXQZIFVVKXSJQHRDZHQYVHKMKFZQYUZKRPFCCYOEHFVQMYZQNRDZWVDZXZN";
const PATTERN_YB: &str = "SBPAZTJFEWEZTNKSMZFFOKZLDXKUYGMIAZRUDZAPQEUQDNOZJBJVGBTDSZVUVHZSASGCAMPZTJKZSRPOSHGBZNAWHZTQSZWYXQHAQWHFJLNFQJXZNHCUUZPZFLANLOJCRZVVAZUEFWKBNZLXJAZHQIOCGITTOFFWQUFVQEIVZWIYAPNDGWYSTLULUAFESXAAMEAPZDHHEGASAKBMQEOFWZIUFBSWSRQZYAZRTZOOOAARLKLXEMLFSXSSYHBWRLLC";

fn decode_pattern(encoded: &str) -> [i8; 256] {
    let bytes = encoded.as_bytes();
    assert_eq!(bytes.len(), 256, "pinned ORB pattern length changed");
    let mut result = [0_i8; 256];
    for (index, &byte) in bytes.iter().enumerate() {
        result[index] = byte as i8 - b'A' as i8 - 13;
    }
    result
}

fn descriptor_pattern() -> &'static DescriptorPattern {
    static PATTERN: OnceLock<DescriptorPattern> = OnceLock::new();
    PATTERN.get_or_init(|| DescriptorPattern {
        xa: decode_pattern(PATTERN_XA),
        ya: decode_pattern(PATTERN_YA),
        xb: decode_pattern(PATTERN_XB),
        yb: decode_pattern(PATTERN_YB),
    })
}

#[inline(always)]
fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut distance = 0;
    for offset in (0..32).step_by(8) {
        let left = u64::from_le_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let right = u64::from_le_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);
        distance += (left ^ right).count_ones();
    }
    distance
}

#[cfg(test)]
mod tests {
    /// A grey image with one bright square (strong corners) at the left and
    /// faint dots (weak corners) over the right half.
    fn lamp_and_texture() -> crate::pyramid::RawU16Image {
        crate::pyramid::RawU16Image::from_fn(160, 120, |x, y| {
            let lamp = (30..50).contains(&x) && (50..70).contains(&y);
            let dot = x >= 80 && x % 12 == 0 && y % 12 == 0;
            let level: u16 = if lamp {
                250
            } else if dot {
                110
            } else {
                100
            };
            level << 8
        })
        .expect("image")
    }

    #[test]
    fn a_lower_quality_level_reaches_weak_texture() {
        use super::{
            detect_keypoints_mapping, detect_keypoints_mapping_with, MAPPING_QUALITY_LEVEL,
        };
        let image = lamp_and_texture();
        let upstream = detect_keypoints_mapping(&image, 50);
        assert_eq!(
            detect_keypoints_mapping_with(&image, 50, MAPPING_QUALITY_LEVEL),
            upstream
        );
        // The lamp's corners set the bar: the faint dots stay below 1% of it.
        let right =
            |points: &[nalgebra::Point2<f64>]| points.iter().filter(|p| p.x >= 80.0).count();
        assert_eq!(right(&upstream), 0, "{upstream:?}");
        // At 0.1% they come in.
        let low = detect_keypoints_mapping_with(&image, 50, 0.001);
        assert!(right(&low) >= 10, "{low:?}");
    }

    /// The detector written pixel by pixel, every read reflected, and the
    /// distance checked against every chosen point.
    fn direct_detection(
        image: &crate::pyramid::RawU16Image,
        num_features: usize,
        quality_level: f32,
    ) -> Vec<nalgebra::Point2<f64>> {
        use super::{reflect101, EDGE_THRESHOLD};
        let (width, height) = (image.width() as i32, image.height() as i32);
        let at = |values: &[f32], x: i32, y: i32| {
            values[(reflect101(y, height) * width + reflect101(x, width)) as usize]
        };
        let pass = |values: &[f32], f: &dyn Fn(&[f32], i32, i32) -> f32| {
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| f(values, x, y))
                .collect::<Vec<_>>()
        };
        let gray = pass(&[], &|_, x, y| {
            f32::from(image.pixel(x as usize, y as usize).unwrap() >> 8)
        });
        let scale = 1.0_f32 / (4.0 * 3.0 * 255.0);
        let dx_h = pass(&gray, &|v, x, y| -at(v, x - 1, y) + at(v, x + 1, y));
        let dy_h = pass(&gray, &|v, x, y| {
            at(v, x - 1, y) + 2.0 * at(v, x, y) + at(v, x + 1, y)
        });
        let dx = pass(&dx_h, &|v, x, y| {
            (at(v, x, y - 1) + 2.0 * at(v, x, y) + at(v, x, y + 1)) * scale
        });
        let dy = pass(&dy_h, &|v, x, y| {
            (-at(v, x, y - 1) + at(v, x, y + 1)) * scale
        });
        let product =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).collect::<Vec<_>>();
        let sums = [product(&dx, &dx), product(&dx, &dy), product(&dy, &dy)].map(|c| {
            let row = pass(&c, &|v, x, y| {
                at(v, x - 1, y) + at(v, x, y) + at(v, x + 1, y)
            });
            pass(&row, &|v, x, y| {
                at(v, x, y - 1) + at(v, x, y) + at(v, x, y + 1)
            })
        });
        let response = (0..sums[0].len())
            .map(|i| {
                let (a, b, c) = (sums[0][i] * 0.5, sums[1][i], sums[2][i] * 0.5);
                (a + c) - ((a - c) * (a - c) + b * b).sqrt()
            })
            .collect::<Vec<_>>();
        let quality = response.iter().copied().fold(0.0_f32, f32::max) * quality_level;
        let mut candidates = Vec::new();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let value = at(&response, x, y);
                let peak = (-1..=1)
                    .flat_map(|j| (-1..=1).map(move |i| (i, j)))
                    .all(|(i, j)| value >= at(&response, x + i, y + j));
                if value >= quality && peak {
                    candidates.push((value, x, y));
                }
            }
        }
        candidates.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| b.1.cmp(&a.1))
        });
        let mut points = Vec::<nalgebra::Point2<f64>>::new();
        for (_, x, y) in candidates {
            if points.len() < num_features
                && points.iter().all(|p| {
                    let (dx, dy) = (p.x as f32 - x as f32, p.y as f32 - y as f32);
                    dx * dx + dy * dy >= 64.0
                })
            {
                points.push(nalgebra::Point2::new(f64::from(x), f64::from(y)));
            }
        }
        let edge =
            |v: f64, size: i32| v >= EDGE_THRESHOLD && v < f64::from(size) - EDGE_THRESHOLD - 1.0;
        points
            .into_iter()
            .filter(|p| edge(p.x, width) && edge(p.y, height))
            .collect()
    }

    #[test]
    fn detection_matches_the_direct_computation() {
        use super::detect_keypoints_mapping_with;
        // Texture at many contrasts, on an odd size.
        let image = crate::pyramid::RawU16Image::from_fn(131, 97, |x, y| {
            let value = (x * 37 + y * 101 + (x * y) % 13 * 29) % 251;
            (value as u16 * (1 + (x / 24) as u16) / 4) << 8
        })
        .expect("image");
        for (num_features, quality) in [(60, 0.01), (400, 0.001), (12, 0.2)] {
            let fast = detect_keypoints_mapping_with(&image, num_features, quality);
            assert!(!fast.is_empty());
            assert_eq!(fast, direct_detection(&image, num_features, quality));
        }
    }

    #[test]
    fn stewenius_schur_rejects_nonfinite_inputs() {
        use nalgebra::Complex;
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for imaginary in [false, true] {
                let mut matrix = super::ComplexMatrix10::identity();
                matrix[(3, 7)] = if imaginary {
                    Complex::new(0.0, invalid)
                } else {
                    Complex::new(invalid, 0.0)
                };
                assert!(super::stewenius_schur_bounded(matrix, 10_000).is_none());
                assert!(super::stewenius_eigenvectors(matrix).is_empty());
            }
        }
    }

    fn finite_schur_fixture(seed: u32) -> super::ComplexMatrix10 {
        let mut state = seed;
        super::ComplexMatrix10::from_fn(|row, col| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = (state >> 8) as f64 / 16777216.0 - 0.5;
            nalgebra::Complex::new(value, if row == col { 0.125 } else { 0.0 })
        })
    }

    #[test]
    fn stewenius_schur_preserves_convergent_decomposition_bits() {
        for seed in 1..=64 {
            let matrix = finite_schur_fixture(seed);
            let expected = nalgebra::linalg::Schur::new(matrix).unpack();
            let actual =
                super::stewenius_schur_bounded(matrix, super::STEWENIUS_SCHUR_MAX_ITERATIONS)
                    .expect("finite fixture must converge within the budget");
            for (left, right) in actual
                .0
                .iter()
                .chain(actual.1.iter())
                .zip(expected.0.iter().chain(expected.1.iter()))
            {
                assert_eq!(left.re.to_bits(), right.re.to_bits(), "seed={seed}");
                assert_eq!(left.im.to_bits(), right.im.to_bits(), "seed={seed}");
            }
        }
    }

    #[test]
    fn stewenius_schur_reports_iteration_exhaustion() {
        assert!(super::stewenius_schur_bounded(finite_schur_fixture(7), 1).is_none());
    }

    use super::{
        compute_angles, compute_descriptors, descriptor_pattern, opengv_cayley2rot,
        opengv_eigen_dot3, opengv_eigen_mat34_vec4, opengv_eigen_matvec3, opengv_eigen_norm3,
        opengv_optimize_pose, opengv_qrsolv, opengv_rot2cayley, relative_pose_error,
        stewenius_eigen_col_piv_qr_f64, stewenius_eigen_full_v_last4_f64, triangulate2,
        DescriptorMatch, OpenGvMt19937, RelativePoseModel,
    };
    use crate::pyramid::RawU16Image;
    use nalgebra::{DMatrix, DVector, Matrix3, Point2, SMatrix, Vector3, Vector4};
    use serde_json::Value;

    #[test]
    #[ignore = "requires external native Stewenius QR capture on E"]
    fn m11_stewenius_col_piv_qr_oracle() {
        let path = std::env::var("M11_STEWENIUS_NATIVE_JSON").expect("native oracle path");
        let oracle: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let q_json = oracle["stewenius_q"].as_array().unwrap();
        let mut q = DMatrix::<f64>::zeros(q_json.len(), 9);
        for row in 0..q_json.len() {
            for column in 0..9 {
                q[(row, column)] = q_json[row][column].as_f64().unwrap();
            }
        }
        let (qr, h, cols, matrix_q) = stewenius_eigen_col_piv_qr_f64(&q);
        let expected_qr = oracle["stewenius_qr_matrix"].as_array().unwrap();
        let expected_h = oracle["stewenius_qr_h"].as_array().unwrap();
        let expected_cols = oracle["stewenius_qr_cols_permutation"].as_array().unwrap();
        let expected_q = oracle["stewenius_qr_q"].as_array().unwrap();
        let mut qr_max = 0.0_f64;
        let mut q_max = 0.0_f64;
        for row in 0..9 {
            for column in 0..q_json.len() {
                qr_max = qr_max
                    .max((qr[(row, column)] - expected_qr[row][column].as_f64().unwrap()).abs());
            }
            for column in 0..9 {
                q_max = q_max.max(
                    (matrix_q[(row, column)] - expected_q[row][column].as_f64().unwrap()).abs(),
                );
            }
        }
        let h_max = h
            .iter()
            .enumerate()
            .map(|(i, value)| (value - expected_h[i].as_f64().unwrap()).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "STEWENIUS_QR qr_max={qr_max:.17e} h_max={h_max:.17e} q_max={q_max:.17e} cols={cols:?}"
        );
        assert_eq!(
            cols,
            expected_cols
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
        let full_v = stewenius_eigen_full_v_last4_f64(&q).unwrap();
        let expected_v = oracle["nullspace_v_last4"].as_array().unwrap();
        let mut v_max = 0.0_f64;
        for row in 0..9 {
            for column in 0..4 {
                v_max = v_max
                    .max((full_v[(row, column)] - expected_v[row][column].as_f64().unwrap()).abs());
            }
        }
        eprintln!("STEWENIUS_FULL_V max_abs={v_max:.17e}");
        assert!(v_max < 1e-12);
    }

    #[test]
    fn opengv_mt19937_uniform_stream_crosses_twist_boundary() {
        let mut rng = OpenGvMt19937::new(12345);
        let mut hash = 1_469_598_103_934_665_603_u64;
        for _ in 0..4096 {
            let value = rng.uniform_int_max();
            for shift in (0..32).step_by(8) {
                hash ^= u64::from((value >> shift) as u8);
                hash = hash.wrapping_mul(1_099_511_628_211);
            }
        }
        // GCC 11.4's std::uniform_int_distribution<int>(0, INT_MAX),
        // including the 624-value mt19937 twist boundary.
        assert_eq!(hash, 4_040_478_949_708_829_921);
    }

    #[test]
    fn relative_pose_from_bearings_recovers_left_from_right_pose() {
        // KAT: exact bearings of points seen from two cameras give back
        // T_left_right (p_left = R p_right + t) with unit t, and every
        // correspondence is an inlier. Pins the direction convention used by
        // the visual-only bootstrap.
        let rotation = nalgebra::Rotation3::from_euler_angles(0.02, -0.05, 0.03).into_inner();
        let translation = Vector3::new(0.3, -0.1, 0.05);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for i in 0..60 {
            let f = i as f64;
            let p_left = Vector3::new(
                (f * 0.37).sin() * 1.5,
                (f * 0.71).cos() * 1.0,
                3.0 + (f * 0.53).sin() * 1.5,
            );
            let p_right = rotation.transpose() * (p_left - translation);
            left.push(p_left.normalize());
            right.push(p_right.normalize());
        }
        let (r, t, inliers) =
            super::relative_pose_from_bearings(&left, &right, 5e-5, 42).expect("model");
        assert_eq!(inliers.len(), 60);
        assert!((r - rotation).abs().max() < 1e-6, "rotation {r}");
        assert!(
            (t - translation.normalize()).norm() < 1e-6,
            "translation {t}"
        );
    }

    /// Regression test for a panic ("Matrix index out of bounds" at
    /// `opengv_eigen_cpqr`'s `qr[(k, k)]`) observed on real (non-EuRoC)
    /// imagery whose two-view RANSAC best-consensus set has fewer than 6
    /// inliers. `opengv_eigen_cpqr` column-pivots a (rows x 6) Jacobian and
    /// indexes row k for k in 0..6, so it needs rows >= 6; `inliers.len()`
    /// becomes the row count. Below 6 inliers, `opengv_optimize_pose` must
    /// return the unrefined `initial` model (matching its existing
    /// zero-inlier early return) instead of calling into the QR at all.
    #[test]
    fn opengv_optimize_pose_skips_refinement_below_six_inliers() {
        let initial = RelativePoseModel {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.1, 0.2, 0.3),
        };
        // Five candidate correspondences: enough to be a non-empty
        // consensus set, but one short of the 6 rows the LM Jacobian needs.
        let left_rays: Vec<[f64; 4]> = (0..5)
            .map(|i| [1.0, 0.1 * i as f64, 0.05 * i as f64, 0.0])
            .collect();
        let right_rays: Vec<[f64; 4]> = left_rays.clone();
        let matches: Vec<DescriptorMatch> = (0..5)
            .map(|i| DescriptorMatch {
                left: i,
                right: i,
                distance: 0,
            })
            .collect();
        let inliers: Vec<usize> = (0..5).collect();

        let refined = opengv_optimize_pose(initial, &left_rays, &right_rays, &matches, &inliers);

        assert_eq!(refined.rotation, initial.rotation);
        assert_eq!(refined.translation, initial.translation);
    }

    #[test]
    fn pinned_descriptor_pattern_is_decoded_and_sampled() {
        let pattern = descriptor_pattern();
        assert_eq!(pattern.xa.len(), 256);
        assert_eq!(pattern.ya.len(), 256);
        assert_eq!(pattern.xb.len(), 256);
        assert_eq!(pattern.yb.len(), 256);

        let image =
            RawU16Image::from_fn(64, 64, |x, y| ((x * 257 + y * 509) & 0xffff) as u16).unwrap();
        let corners = [Point2::new(32.0, 32.0)];
        let angles = compute_angles(&image, &corners, true);
        let descriptors = compute_descriptors(&image, &corners, &angles).unwrap();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].len(), 32);
    }

    #[test]
    fn opengv_qrsolv_remirrors_r_for_each_damping_solve() {
        let mut upper = DMatrix::<f64>::from_fn(6, 6, |row, column| {
            if row > column {
                0.0
            } else if row == column {
                2.0 + row as f64
            } else {
                0.1 * (row + column + 1) as f64
            }
        });
        for row in 0..6 {
            for column in 0..row {
                upper[(row, column)] = upper[(column, row)];
            }
        }
        let pivots = [0, 1, 2, 3, 4, 5];
        let qtb = DVector::from_iterator(6, (0..6).map(|index| 0.3 + index as f64 * 0.2));
        let damping1 = DVector::from_iterator(6, (0..6).map(|index| 0.2 + index as f64 * 0.1));
        let damping2 = DVector::from_iterator(6, (0..6).map(|index| 0.4 + index as f64 * 0.07));
        let mut reused = upper.clone();
        let mut fresh = upper;
        let mut reused_x = DVector::zeros(6);
        let mut reused_sdiag = DVector::zeros(6);
        opengv_qrsolv(
            &mut reused,
            &pivots,
            &damping1,
            &qtb,
            &mut reused_x,
            &mut reused_sdiag,
        );
        opengv_qrsolv(
            &mut reused,
            &pivots,
            &damping2,
            &qtb,
            &mut reused_x,
            &mut reused_sdiag,
        );
        let mut fresh_x = DVector::zeros(6);
        let mut fresh_sdiag = DVector::zeros(6);
        opengv_qrsolv(
            &mut fresh,
            &pivots,
            &damping2,
            &qtb,
            &mut fresh_x,
            &mut fresh_sdiag,
        );
        for index in 0..6 {
            assert!((reused_x[index] - fresh_x[index]).abs() < 1e-12);
            assert!((reused_sdiag[index] - fresh_sdiag[index]).abs() < 1e-12);
        }
    }

    #[test]
    #[ignore]
    fn m8c_debug_fixed_oracle_lm_trace() {
        let raw_path = std::env::var_os("M8C_FEATURE_RAW_JSON")
            .expect("M8C_FEATURE_RAW_JSON for disposable fixed-model trace");
        let raw: Value =
            serde_json::from_str(&std::fs::read_to_string(raw_path).expect("raw fixture"))
                .expect("raw JSON");
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/basalt/m8c_feature_oracle20_ransac.json"
        ))
        .expect("oracle JSON");
        let feature = |frame: u64, cam: u64| {
            raw["features"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| {
                    item["time_cam_id"]["frame_id"].as_u64() == Some(frame)
                        && item["time_cam_id"]["cam_id"].as_u64() == Some(cam)
                })
                .unwrap()
        };
        let left_frame = 1_403_636_580_113_555_456_u64;
        let right_frame = 1_403_636_579_763_555_584_u64;
        let left_feature = feature(left_frame, 0);
        let right_feature = feature(right_frame, 1);
        let rays = |item: &Value| {
            item["corners_3d"]
                .as_array()
                .unwrap()
                .iter()
                .map(|ray| {
                    [
                        ray[0].as_f64().unwrap(),
                        ray[1].as_f64().unwrap(),
                        ray[2].as_f64().unwrap(),
                        ray[3].as_f64().unwrap_or(0.0),
                    ]
                })
                .collect::<Vec<_>>()
        };
        let left_rays = rays(left_feature);
        let right_rays = rays(right_feature);
        let entry = oracle["bow_query"]["seeded_temporal_oracle"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| {
                entry["seed"].as_u64() == Some(7)
                    && entry["left"]["frame_id"].as_u64() == Some(left_frame)
                    && entry["right"]["frame_id"].as_u64() == Some(right_frame)
                    && entry["right"]["cam_id"].as_u64() == Some(1)
            })
            .unwrap();
        let mut matches = Vec::new();
        for pair in entry["ransac_inlier_ids"].as_array().unwrap() {
            matches.push(DescriptorMatch {
                left: pair[0].as_u64().unwrap(),
                right: pair[1].as_u64().unwrap(),
                distance: 0,
            });
        }
        let rotation = Matrix3::from_fn(|row, column| {
            entry["ransac_model_rotation"][row][column]
                .as_f64()
                .unwrap()
        });
        let translation = Vector3::new(
            entry["ransac_model_translation"][0].as_f64().unwrap(),
            entry["ransac_model_translation"][1].as_f64().unwrap(),
            entry["ransac_model_translation"][2].as_f64().unwrap(),
        );
        if std::env::var_os("M8C_FEATURE_PARITY_POINT_TRACE").is_some() {
            let point_trace_count =
                if std::env::var_os("M8C_FEATURE_PARITY_POINT_TRACE_ALL").is_some() {
                    entry["ransac_inlier_ids"].as_array().unwrap().len()
                } else {
                    8
                };
            for (index, pair) in entry["ransac_inlier_ids"]
                .as_array()
                .unwrap()
                .iter()
                .take(point_trace_count)
                .enumerate()
            {
                let left = left_rays[pair[0].as_u64().unwrap() as usize];
                let right = right_rays[pair[1].as_u64().unwrap() as usize];
                let model = RelativePoseModel {
                    rotation: opengv_cayley2rot(opengv_rot2cayley(rotation)),
                    translation,
                };
                let point = triangulate2(
                    Vector3::new(left[0], left[1], left[2]),
                    Vector3::new(right[0], right[1], right[2]),
                    model.rotation,
                    model.translation,
                )
                .unwrap();
                let inverse_rotation = model.rotation.transpose();
                let inverse_translation =
                    -opengv_eigen_matvec3(inverse_rotation, model.translation);
                let mut inverse_solution = SMatrix::<f64, 3, 4>::zeros();
                inverse_solution
                    .fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(&inverse_rotation);
                inverse_solution
                    .column_mut(3)
                    .copy_from(&inverse_translation);
                let point_hom = Vector4::new(point[0], point[1], point[2], 1.0);
                let second = opengv_eigen_mat34_vec4(inverse_solution, point_hom);
                let first_norm = opengv_eigen_norm3(&point);
                let second_norm = opengv_eigen_norm3(&second);
                let first_unit = point / first_norm;
                let second_unit = second / second_norm;
                let first_dot =
                    opengv_eigen_dot3(&Vector3::new(left[0], left[1], left[2]), &first_unit);
                let second_dot =
                    opengv_eigen_dot3(&Vector3::new(right[0], right[1], right[2]), &second_unit);
                let residual = (1.0 - first_dot) + (1.0 - second_dot);
                eprintln!(
                    "M8C_FIXED_POINT index={index} bits={:016x},{:016x},{:016x} second={:016x},{:016x},{:016x} norm={:016x},{:016x} dot={:016x},{:016x} residual={:016x}",
                    point[0].to_bits(),
                    point[1].to_bits(),
                    point[2].to_bits(),
                    second[0].to_bits(),
                    second[1].to_bits(),
                    second[2].to_bits(),
                    first_norm.to_bits(),
                    second_norm.to_bits(),
                    first_dot.to_bits(),
                    second_dot.to_bits(),
                    residual.to_bits(),
                );
            }
        }
        if std::env::var_os("M8C_FEATURE_PARITY_PERT_BITS").is_some() {
            let cayley = opengv_rot2cayley(rotation);
            let mut parameters = [0.0_f64; 6];
            parameters[0] = translation[0];
            parameters[1] = translation[1];
            parameters[2] = translation[2];
            parameters[3] = cayley[0];
            parameters[4] = cayley[1];
            parameters[5] = cayley[2];
            let emit = |label: &str, parameters: &[f64; 6]| {
                let model = RelativePoseModel {
                    rotation: opengv_cayley2rot(Vector3::new(
                        parameters[3],
                        parameters[4],
                        parameters[5],
                    )),
                    translation: Vector3::new(parameters[0], parameters[1], parameters[2]),
                };
                let mut bits = Vec::new();
                let trace_count = if std::env::var_os("M8C_FEATURE_PARITY_PERT_BITS_ALL").is_some()
                {
                    entry["ransac_inlier_ids"].as_array().unwrap().len()
                } else {
                    8
                };
                for pair in entry["ransac_inlier_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .take(trace_count)
                {
                    let left = left_rays[pair[0].as_u64().unwrap() as usize];
                    let right = right_rays[pair[1].as_u64().unwrap() as usize];
                    let residual = relative_pose_error(
                        Vector3::new(left[0], left[1], left[2]),
                        Vector3::new(right[0], right[1], right[2]),
                        model,
                    );
                    bits.push(format!("{:016x}", residual.to_bits()));
                }
                eprintln!("M8C_FIXED_PERT_BITS {label} {}", bits.join(","));
            };
            for column in 0..6 {
                let mut perturbed = parameters;
                let mut h = f64::EPSILON.sqrt() * parameters[column].abs();
                if h == 0.0 {
                    h = f64::EPSILON.sqrt();
                }
                perturbed[column] += h;
                emit(&format!("col={column} h={h:.17e}"), &perturbed);
            }
        }
        let model = opengv_optimize_pose(
            RelativePoseModel {
                rotation,
                translation,
            },
            &left_rays,
            &right_rays,
            &matches,
            &(0..matches.len()).collect::<Vec<_>>(),
        );
        eprintln!("M8C_FIXED_ORACLE_MODEL={model:?}");
    }
}
