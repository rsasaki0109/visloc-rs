//! Dependency-free OpenCV-compatible FAST-9 grid detector for Basalt
//! replenishment.
//!
//! Basalt converts raw `u16` pixels to `u8` with `pixel >> 8`, runs OpenCV
//! FAST in each configured grid cell, and lowers the threshold from 40 to 5
//! until the cell quota is filled. The detector follows the pinned OpenCV
//! 4.12.0 `FAST_t<16>` corner score and NMS, including the three-pixel
//! sub-image border and response-only `std::sort` ordering, without bringing
//! OpenCV or descriptor matching into the direct KLT stream. See
//! `pipelines/basalt/PROVENANCE.md` for the source and BSD-3-Clause notice.

use std::collections::HashSet;

use nalgebra::Vector2;
use rayon::prelude::*;

use crate::pyramid::RawU16Image;

// OpenCV's `makeOffsets` starts at the bottom of the Bresenham circle and
// walks clockwise.  FAST is invariant to a rotation of this ring, so the
// equivalent top-first ordering is used here.  Keeping the ring explicit is
// useful because it makes the integer arithmetic below independent of an
// image-processing runtime.
const FAST_CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Grid and FAST parameters matching Basalt's keypoint replenishment defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridFastConfig {
    pub cell_size: usize,
    pub points_per_cell: usize,
    pub threshold: u8,
    pub min_threshold: u8,
    pub edge_threshold: usize,
}

impl Default for GridFastConfig {
    fn default() -> Self {
        Self {
            cell_size: 50,
            points_per_cell: 1,
            threshold: 40,
            min_threshold: 5,
            edge_threshold: 19,
        }
    }
}

/// Deterministic FAST-9 detector used only for cam0 grid replenishment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridFastDetector {
    pub config: GridFastConfig,
}

impl Default for GridFastDetector {
    fn default() -> Self {
        Self {
            config: GridFastConfig::default(),
        }
    }
}

impl GridFastDetector {
    pub const fn new(config: GridFastConfig) -> Self {
        Self { config }
    }

    /// Detects at most `points_per_cell` points in each currently unoccupied
    /// grid cell. Existing points are supplied in full-resolution pixel
    /// coordinates, as in Basalt's `detectKeypoints` call.
    pub fn detect(&self, image: &RawU16Image, existing: &[Vector2<f32>]) -> Vec<Vector2<f32>> {
        let cell = self.config.cell_size;
        if cell == 0 || self.config.points_per_cell == 0 {
            return Vec::new();
        }
        let cells_x = image.width() / cell;
        let cells_y = image.height() / cell;
        if cells_x == 0 || cells_y == 0 {
            return Vec::new();
        }

        let x_start = (image.width() % cell) / 2;
        let y_start = (image.height() % cell) / 2;
        let mut occupied = HashSet::new();
        for point in existing {
            if point.x < x_start as f32
                || point.y < y_start as f32
                || point.x >= (x_start + cells_x * cell) as f32
                || point.y >= (y_start + cells_y * cell) as f32
            {
                continue;
            }
            occupied.insert((
                ((point.x as usize) - x_start) / cell,
                ((point.y as usize) - y_start) / cell,
            ));
        }

        // Each unoccupied cell's threshold-escalating FAST-9 search is a
        // pure function of that cell's own `(x0, y0)` sub-image plus the
        // shared, read-only `image` and `self.config` -- no cell reads or
        // writes another cell's `score_grid` or candidate list. Computing
        // every cell's point(s) in parallel and then extending `result`
        // serially, in the exact same `cell_x` outer / `cell_y` inner
        // visitation order the sequential nested loop always used,
        // reproduces the identical `result` vector (and therefore the
        // identical downstream new-track-id assignment order in the
        // frontend) regardless of thread count; only *when* each
        // independent cell's search runs changes, never its own candidate
        // scoring, sort, or edge-threshold filtering.
        let cell_coords: Vec<(usize, usize)> = (0..cells_x)
            .flat_map(|cell_x| (0..cells_y).map(move |cell_y| (cell_x, cell_y)))
            .filter(|coords| !occupied.contains(coords))
            .collect();
        let per_cell_points: Vec<Vec<Vector2<f32>>> = cell_coords
            .par_iter()
            .map(|&(cell_x, cell_y)| {
                let x0 = x_start + cell_x * cell;
                let y0 = y_start + cell_y * cell;
                self.detect_cell(image, x0, y0)
            })
            .collect();
        let mut result = Vec::new();
        for points in per_cell_points {
            result.extend(points);
        }
        result
    }

    /// Threshold-escalating FAST-9 search over one grid cell's `cell x cell`
    /// sub-image at `(x0, y0)`. Pure and side-effect-free: every mutable
    /// scratch buffer (`score_grid`, `candidates`) is local to this call, so
    /// many cells can run this concurrently with no coordination.
    fn detect_cell(&self, image: &RawU16Image, x0: usize, y0: usize) -> Vec<Vector2<f32>> {
        let cell = self.config.cell_size;
        let mut result = Vec::new();
        let mut threshold = self.config.threshold;
        let mut selected = 0;
        // OpenCV's FAST_t computes one score row and performs NMS
        // against the retained neighbouring score rows.  Cache the
        // complete cell here as the equivalent bounded form instead
        // of recomputing as many as eight neighbour scores for every
        // candidate pixel.
        let mut score_grid = vec![0_i16; cell * cell];
        while selected < self.config.points_per_cell && threshold >= self.config.min_threshold {
            let mut candidates = Vec::new();
            // `cv::FAST` never examines the three-pixel border of its
            // *input image*.  Basalt passes a 50x50 sub-image, so
            // this is a cell-local border, in addition to Basalt's
            // full-image EDGE_THRESHOLD check below.
            if cell <= 6 {
                break;
            }
            score_grid.fill(0);
            for y in y0 + 3..y0 + cell - 3 {
                for x in x0 + 3..x0 + cell - 3 {
                    if let Some(score) =
                        fast_score_without_nms(image, x, y, threshold, x0, y0, cell)
                    {
                        score_grid[(y - y0) * cell + (x - x0)] = score;
                    }
                }
            }
            for y in y0 + 3..y0 + cell - 3 {
                for x in x0 + 3..x0 + cell - 3 {
                    let local_x = x - x0;
                    let local_y = y - y0;
                    let score = score_grid[local_y * cell + local_x];
                    if score == 0 {
                        continue;
                    }
                    let mut is_maximum = true;
                    for dy in -1_i32..=1 {
                        for dx in -1_i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let neighbour_x = (local_x as i32 + dx) as usize;
                            let neighbour_y = (local_y as i32 + dy) as usize;
                            if score_grid[neighbour_y * cell + neighbour_x] >= score {
                                is_maximum = false;
                                break;
                            }
                        }
                        if !is_maximum {
                            break;
                        }
                    }
                    if is_maximum {
                        candidates.push((score as u8, x, y));
                    }
                }
            }
            // Basalt sorts the `cv::FAST` output solely by response.
            // `std::sort` is intentionally not stable; use the small
            // libstdc++-compatible introsort below so equal-response
            // points retain the same implementation-defined order as
            // the pinned GCC/OpenCV oracle.
            opencv_sort_by_response(&mut candidates);

            for (_, x, y) in candidates {
                if selected >= self.config.points_per_cell {
                    break;
                }
                // Upstream filters the full-image EDGE_THRESHOLD
                // only after cv::FAST's response-only sort. Keeping
                // out-of-bounds candidates in the sorted input is
                // observable when equal responses straddle the edge.
                if !image.in_bounds(
                    Vector2::new(x as f32, y as f32),
                    self.config.edge_threshold as f32,
                ) {
                    continue;
                }
                result.push(Vector2::new(x as f32, y as f32));
                selected += 1;
            }
            if threshold == self.config.min_threshold {
                break;
            }
            // This is Basalt's literal `threshold /= 2`.  In
            // particular, do not clamp back up to min_threshold: a
            // threshold of 6 runs once and then terminates at 3, just
            // as the upstream loop does for its fixed lower bound 5.
            threshold /= 2;
        }
        result
    }
}

/// Sorts FAST candidates like the libstdc++ `std::sort` used by Basalt's
/// `keypoints.cpp` (GCC 11.4, OpenCV 4.12.0).  Rust's stable sort and its
/// pdqsort-based unstable sort intentionally make different choices for
/// equivalent elements; those choices affect which point wins a one-point
/// grid cell when responses tie.
fn opencv_sort_by_response(candidates: &mut [(u8, usize, usize)]) {
    if candidates.len() < 2 {
        return;
    }

    let depth_limit = 2 * floor_log2(candidates.len());
    introsort_loop(candidates, depth_limit);
    final_insertion_sort(candidates);
}

const fn floor_log2(value: usize) -> usize {
    (usize::BITS - 1 - value.leading_zeros()) as usize
}

const fn less_response(a: &(u8, usize, usize), b: &(u8, usize, usize)) -> bool {
    a.0 > b.0
}

fn introsort_loop(mut candidates: &mut [(u8, usize, usize)], mut depth_limit: usize) {
    const INSERTION_THRESHOLD: usize = 16;
    while candidates.len() > INSERTION_THRESHOLD {
        if depth_limit == 0 {
            // This fallback is practically unreachable for FAST candidate
            // lists, but preserves std::sort's total-order guarantee.
            candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(candidates);
        introsort_loop(&mut candidates[cut..], depth_limit);
        candidates = &mut candidates[..cut];
    }
}

fn move_median_to_first(candidates: &mut [(u8, usize, usize)], a: usize, b: usize, c: usize) {
    if less_response(&candidates[a], &candidates[b]) {
        if less_response(&candidates[b], &candidates[c]) {
            candidates.swap(0, b);
        } else if less_response(&candidates[a], &candidates[c]) {
            candidates.swap(0, c);
        } else {
            candidates.swap(0, a);
        }
    } else if less_response(&candidates[a], &candidates[c]) {
        candidates.swap(0, a);
    } else if less_response(&candidates[b], &candidates[c]) {
        candidates.swap(0, c);
    } else {
        candidates.swap(0, b);
    }
}

fn unguarded_partition_pivot(candidates: &mut [(u8, usize, usize)]) -> usize {
    let middle = candidates.len() / 2;
    let last = candidates.len() - 1;
    move_median_to_first(candidates, 1, middle, last);

    let mut first = 1;
    let mut last = candidates.len();
    loop {
        while less_response(&candidates[first], &candidates[0]) {
            first += 1;
        }
        last -= 1;
        while less_response(&candidates[0], &candidates[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        candidates.swap(first, last);
        first += 1;
    }
}

fn final_insertion_sort(candidates: &mut [(u8, usize, usize)]) {
    const INSERTION_THRESHOLD: usize = 16;
    if candidates.len() > INSERTION_THRESHOLD {
        insertion_sort(&mut candidates[..INSERTION_THRESHOLD]);
        for index in INSERTION_THRESHOLD..candidates.len() {
            unguarded_linear_insert(candidates, index);
        }
    } else {
        insertion_sort(candidates);
    }
}

fn insertion_sort(candidates: &mut [(u8, usize, usize)]) {
    for index in 1..candidates.len() {
        if less_response(&candidates[index], &candidates[0]) {
            let value = candidates[index];
            candidates.copy_within(0..index, 1);
            candidates[0] = value;
        } else {
            unguarded_linear_insert(candidates, index);
        }
    }
}

fn unguarded_linear_insert(candidates: &mut [(u8, usize, usize)], index: usize) {
    let value = candidates[index];
    let mut position = index;
    while position > 0 && less_response(&value, &candidates[position - 1]) {
        candidates[position] = candidates[position - 1];
        position -= 1;
    }
    candidates[position] = value;
}

#[cfg(test)]
fn fast_score(
    image: &RawU16Image,
    x: usize,
    y: usize,
    threshold: u8,
    _edge_threshold: usize,
    cell_x: usize,
    cell_y: usize,
    cell_size: usize,
) -> Option<u8> {
    if !fast_pixel_in_cell(x, y, cell_x, cell_y, cell_size) {
        return None;
    }
    let score = fast_score_without_nms(image, x, y, threshold, cell_x, cell_y, cell_size)?;

    // OpenCV's FAST_t stores the score for a candidate and accepts it only if
    // it is strictly greater than all eight adjacent scores.  Pixels outside
    // the cell-local FAST scan have score zero (rather than being evaluated as
    // neighbors), which matters at a sub-image boundary.
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 {
                if let Some(neighbour) = fast_score_without_nms(
                    image,
                    nx as usize,
                    ny as usize,
                    threshold,
                    cell_x,
                    cell_y,
                    cell_size,
                ) {
                    // NMS is strict (`score > neighbour` in OpenCV), so an
                    // equal-score neighbour suppresses both candidates.
                    if neighbour >= score {
                        return None;
                    }
                }
            }
        }
    }
    Some(score as u8)
}

/// Returns true for the pixels examined by OpenCV's FAST_t implementation on
/// a `cell_size x cell_size` `cv::Mat` sub-image.  Its scan starts at 3 and
/// stops before `size - 3`, because the 16-pixel circle has radius 3.
const fn fast_pixel_in_cell(
    x: usize,
    y: usize,
    cell_x: usize,
    cell_y: usize,
    cell_size: usize,
) -> bool {
    x >= cell_x + 3
        && x < cell_x + cell_size.saturating_sub(3)
        && y >= cell_y + 3
        && y < cell_y + cell_size.saturating_sub(3)
}

/// Computes OpenCV's FAST-9 corner score without non-maximum suppression.
///
/// OpenCV's `cornerScore<16>` uses `d[k] = center - ring[k]` and returns one
/// less than the strongest nine-pixel contiguous contrast, considering both
/// dark and bright arcs.  The direct formulation below is algebraically the
/// same as the pinned 4.12.0 implementation's scalar/SIMD reduction, while
/// making the strict threshold comparison explicit at the call site.
fn fast_score_without_nms(
    image: &RawU16Image,
    x: usize,
    y: usize,
    threshold: u8,
    cell_x: usize,
    cell_y: usize,
    cell_size: usize,
) -> Option<i16> {
    if !fast_pixel_in_cell(x, y, cell_x, cell_y, cell_size) {
        return None;
    }

    let center = raw_to_u8(image.pixel(x, y)?) as i16;
    let threshold = i16::from(threshold);
    // A run of 9 of the 16 circle pixels always holds two neighbouring
    // compass pixels (indices 0, 4, 8, 12), so a pixel with no such pair on
    // the same side of the threshold has no run: OpenCV's quick rejection,
    // exact, and it skips the other twelve reads for most pixels.
    let mut compass_bright = 0_u8;
    let mut compass_dark = 0_u8;
    for (bit, index) in [0, 4, 8, 12].into_iter().enumerate() {
        let (dx, dy) = FAST_CIRCLE[index];
        let (px, py) = (x as i32 + dx, y as i32 + dy);
        if px < 0 || py < 0 {
            return None;
        }
        let difference = center - raw_to_u8(image.pixel(px as usize, py as usize)?) as i16;
        if difference < -threshold {
            compass_bright |= 1 << bit;
        }
        if difference > threshold {
            compass_dark |= 1 << bit;
        }
    }
    let neighbouring_pair = |mask: u8| mask & (mask >> 1 | mask << 3) & 0xf != 0;
    if !neighbouring_pair(compass_bright) && !neighbouring_pair(compass_dark) {
        return None;
    }
    let mut differences = [0_i16; 24];
    let mut bright_mask = 0_u16;
    let mut dark_mask = 0_u16;
    for (index, (dx, dy)) in FAST_CIRCLE.iter().enumerate() {
        let px = x as i32 + dx;
        let py = y as i32 + dy;
        if px < 0 || py < 0 {
            return None;
        }
        let difference = center - raw_to_u8(image.pixel(px as usize, py as usize)?) as i16;
        differences[index] = difference;
        if difference < -threshold {
            bright_mask |= 1 << index;
        }
        if difference > threshold {
            dark_mask |= 1 << index;
        }
    }

    if !has_circular_run9(bright_mask) && !has_circular_run9(dark_mask) {
        return None;
    }

    let (head, wrap) = differences.split_at_mut(16);
    wrap.copy_from_slice(&head[..8]);
    let mut score = i16::MIN;
    for start in 0..16 {
        let mut min_difference = i16::MAX;
        let mut max_difference = i16::MIN;
        for offset in 0..9 {
            let difference = differences[start + offset];
            min_difference = min_difference.min(difference);
            max_difference = max_difference.max(difference);
        }
        // `min_difference - 1` is the score for a dark arc; the negated
        // maximum is the score for a bright arc.
        score = score.max(min_difference - 1);
        score = score.max(-max_difference - 1);
    }
    Some(score)
}

#[inline]
fn has_circular_run9(mask: u16) -> bool {
    let doubled = u32::from(mask) | (u32::from(mask) << 16);
    let mut starts = doubled;
    for shift in 1..9 {
        starts &= doubled >> shift;
    }
    starts & 0xffff != 0
}

#[inline]
const fn raw_to_u8(pixel: u16) -> u8 {
    // Basalt's dataset reader promotes an 8-bit PNG sample to u16 by shifting
    // left eight bits, and detectKeypointsMapping reverses that conversion
    // with this exact truncating shift (not normalization or rounding).
    (pixel >> 8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_u16_conversion_is_the_pinned_truncating_shift() {
        assert_eq!(raw_to_u8(0x0000), 0x00);
        assert_eq!(raw_to_u8(0x12ff), 0x12);
        assert_eq!(raw_to_u8(0xffff), 0xff);
    }

    #[test]
    fn fast9_score_matches_open_cv_corner_score() {
        let image = RawU16Image::from_fn(50, 50, |x, y| {
            if x == 25 && y == 25 {
                10 << 8
            } else {
                let is_ring = FAST_CIRCLE.iter().enumerate().any(|(index, (dx, dy))| {
                    index < 9 && x as i32 == 25 + dx && y as i32 == 25 + dy
                });
                if is_ring {
                    100 << 8
                } else {
                    10 << 8
                }
            }
        })
        .unwrap();
        assert_eq!(
            fast_score_without_nms(&image, 25, 25, 40, 0, 0, 50),
            Some(89)
        );
        assert_eq!(fast_score(&image, 25, 25, 40, 3, 0, 0, 50), Some(89));

        // A FAST candidate whose minimum contrast is threshold + 1 has a
        // corner score exactly equal to threshold; OpenCV still keeps it.
        let threshold_edge =
            RawU16Image::from_fn(50, 50, |x, y| {
                if FAST_CIRCLE.iter().enumerate().any(|(index, (dx, dy))| {
                    index < 9 && x as i32 == 25 + dx && y as i32 == 25 + dy
                }) {
                    16 << 8
                } else {
                    10 << 8
                }
            })
            .unwrap();
        assert_eq!(
            fast_score_without_nms(&threshold_edge, 25, 25, 5, 0, 0, 50),
            Some(5)
        );
    }

    #[test]
    fn circular_run9_mask_matches_direct_arc_scan() {
        for mask in 0_u16..=u16::MAX {
            let direct = (0..16)
                .any(|start| (0..9).all(|offset| mask & (1 << ((start + offset) % 16)) != 0));
            assert_eq!(has_circular_run9(mask), direct, "mask={mask:04x}");
        }
    }

    #[test]
    fn opencv_sort_reproduces_equal_response_ordering() {
        let mut candidates = vec![
            (77, 15, 32),
            (77, 18, 23),
            (69, 7, 21),
            (67, 5, 32),
            (60, 32, 30),
            (59, 35, 34),
            (58, 42, 46),
            (58, 32, 40),
            (58, 25, 36),
            (58, 4, 26),
            (57, 40, 27),
            (55, 37, 38),
            (54, 38, 30),
            (54, 18, 8),
            (54, 33, 7),
            (54, 39, 5),
            (52, 45, 29),
            (51, 15, 18),
            (51, 40, 43),
            (51, 21, 42),
            (49, 19, 6),
            (48, 13, 13),
            (46, 10, 41),
            (46, 15, 39),
            (45, 9, 26),
            (45, 12, 45),
            (44, 43, 37),
            (44, 19, 41),
            (44, 16, 27),
            (44, 27, 4),
            (43, 11, 29),
            (42, 5, 40),
            (42, 23, 6),
        ];
        // cv::FAST emits candidates in row-major order before Basalt's
        // response-only std::sort call.
        candidates.sort_by_key(|(_, x, y)| (*y, *x));
        opencv_sort_by_response(&mut candidates);
        assert_eq!(candidates[0], (77, 15, 32));
        assert_eq!(candidates[1], (77, 18, 23));
    }

    #[test]
    fn grid_fast_finds_a_high_contrast_corner_without_descriptors() {
        let image =
            RawU16Image::from_fn(64, 64, |x, y| {
                if FAST_CIRCLE.iter().enumerate().any(|(index, (dx, dy))| {
                    index < 9 && x as i32 == 32 + dx && y as i32 == 32 + dy
                }) {
                    65_535
                } else {
                    0
                }
            })
            .unwrap();
        let detector = GridFastDetector::new(GridFastConfig {
            cell_size: 64,
            points_per_cell: 1,
            threshold: 40,
            min_threshold: 5,
            edge_threshold: 3,
        });
        let points = detector.detect(&image, &[]);
        assert!(!points.is_empty());
        assert!(points
            .iter()
            .any(|p| (p.x - 32.0).abs() < 2.0 && (p.y - 32.0).abs() < 2.0));
    }

    #[test]
    fn cached_score_grid_matches_direct_neighbour_recomputation() {
        let cell = 50;
        let image = RawU16Image::from_fn(cell, cell, |x, y| {
            let value = (x * 37 + y * 61 + x * y * 3 + (x ^ y) * 11) & 0xff;
            (value as u16) << 8
        })
        .unwrap();

        for threshold in [40_u8, 20, 10, 5] {
            let mut score_grid = vec![0_i16; cell * cell];
            for y in 3..cell - 3 {
                for x in 3..cell - 3 {
                    if let Some(score) = fast_score_without_nms(&image, x, y, threshold, 0, 0, cell)
                    {
                        score_grid[y * cell + x] = score;
                    }
                }
            }

            for y in 3..cell - 3 {
                for x in 3..cell - 3 {
                    let direct = fast_score(&image, x, y, threshold, 19, 0, 0, cell);
                    let score = score_grid[y * cell + x];
                    let cached = (score != 0
                        && (-1_i32..=1).all(|dy| {
                            (-1_i32..=1).all(|dx| {
                                (dx == 0 && dy == 0)
                                    || score_grid
                                        [(y as i32 + dy) as usize * cell + (x as i32 + dx) as usize]
                                        < score
                            })
                        }))
                    .then_some(score as u8);
                    assert_eq!(cached, direct, "threshold={threshold} x={x} y={y}");
                }
            }
        }
    }
}
