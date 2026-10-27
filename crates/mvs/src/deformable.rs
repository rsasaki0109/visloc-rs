//! APD-MVS's adaptive patch deformation (Y. Wang et al., "Adaptive Patch
//! Deformation for Textureless-Resilient Multi-View Stereo", CVPR 2023),
//! implemented from the paper and its released code.
//!
//! A pass sorts its pixels at its end, for the next pass (APD's schedule;
//! a finer scale starts from the coarser scale's classes): a pixel is
//! *strong* when its cost along its ray (normal fixed, one disparity step
//! apart) has one clear minimum near its depth, *weak* otherwise. A weak
//! pixel looks for strong *anchors* in sectors around it, far if need be,
//! keeps those RANSAC finds coplanar with a triangle around it, and is
//! matched with its window plus a small window at each anchor, all warped
//! by its own plane: a plane wins only if extended to the textured anchors
//! it matches there. Its candidates are the anchors' planes and a plane
//! fitted to their depths.

use std::collections::VecDeque;

/// Settings of the deformable patches (APD-MVS's code by default).
#[derive(Debug, Clone, PartialEq)]
pub struct Deformable {
    /// Disparity steps sampled on either side of a pixel's depth to sort it.
    pub sweep: i32,
    /// A pixel's lowest cost minimum must lie within this many steps of its
    /// depth (APD: 6 in the first pass at a scale, then 4, 2, 2).
    pub peak_radius: i32,
    /// ... and cost at most this.
    pub max_peak_cost: f32,
    /// A single minimum makes the pixel strong if it costs at most this.
    pub strong_cost: f32,
    /// With several minima, the others' root mean square excess over the
    /// lowest must exceed this.
    pub peak_spread: f32,
    /// Sectors searched for anchors (APD: 16 at the middle scale, 32 at the
    /// finest).
    pub sectors: usize,
    /// Window radius (pixels) searched for a weak sample's nearest strong
    /// pixel.
    pub nearest_radius: usize,
    /// RANSAC inlier distance to the plane, as a fraction of the view's
    /// depth range (APD: 0.875% at the middle scale, 0.75% at the finest).
    pub ransac_threshold: f32,
    /// Coplanar anchors needed (APD: 6).
    pub min_inliers: usize,
    /// A weak pixel's cost: this times its window's plus the rest times the
    /// anchors' mean (APD: 0.25).
    pub center_weight: f32,
    /// After each deformable pass, moves every pixel's depth to the lowest
    /// cost within `LOCAL_REFINE_STEPS` disparity steps along its ray
    /// (normal fixed) if that beats its own by `LOCAL_REFINE_GAIN`
    /// (APD's `LocalRefine`). CPU only.
    pub local_refine: bool,
    /// In a deformable pass, the median filter skips weak pixels and takes
    /// the median of a pixel and its strong neighbours only, over ACMM's
    /// 20-pixel pattern (APD's `CheckerboardFilterStrong`), instead of the
    /// 5×5 median of every depth.
    pub strong_median: bool,
}

/// Disparity steps on either side that `Deformable::local_refine` tries.
pub(crate) const LOCAL_REFINE_STEPS: i32 = 5;

/// Cost gain `Deformable::local_refine` needs to move a depth.
pub(crate) const LOCAL_REFINE_GAIN: f32 = 0.1;

impl Default for Deformable {
    fn default() -> Self {
        Self {
            sweep: 30,
            peak_radius: 4,
            max_peak_cost: 0.5,
            strong_cost: 0.15,
            peak_spread: 0.2,
            sectors: 16,
            nearest_radius: 100,
            ransac_threshold: 0.0075,
            min_inliers: 6,
            center_weight: 0.25,
            local_refine: false,
            strong_median: false,
        }
    }
}

/// Anchors per weak pixel, at most this many.
pub(crate) const MAX_ANCHORS: usize = 8;

/// Pixels this close to the border are never sorted or used as anchors.
const BORDER: usize = 6;

/// How a deformable pass matches a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Matched with its anchors if it has enough coplanar ones, else as a
    /// strong pixel.
    Weak,
    /// Matched with its own window; a possible anchor.
    Strong,
    /// No depth or at the border: matched as a strong pixel, never an
    /// anchor.
    Unknown,
}

/// The class of a pixel from its cost curve `costs` (index `sweep` at its
/// depth; out-of-range depths cost the maximum).
pub(crate) fn classify(costs: &[f32], options: &Deformable) -> Class {
    let center = (costs.len() / 2) as i32;
    let mut peaks = Vec::new();
    for i in 2..costs.len().saturating_sub(2) {
        if costs[i - 1] > costs[i] && costs[i + 1] > costs[i] {
            peaks.push(i);
        }
    }
    let Some(&lowest) = peaks
        .iter()
        .min_by(|&&a, &&b| costs[a].total_cmp(&costs[b]))
    else {
        return Class::Weak;
    };
    let low = costs[lowest];
    if (lowest as i32 - center).abs() > options.peak_radius || low > options.max_peak_cost {
        return Class::Weak;
    }
    if peaks.len() == 1 {
        return if low <= options.strong_cost {
            Class::Strong
        } else {
            Class::Weak
        };
    }
    let excess: f32 = peaks
        .iter()
        .filter(|&&i| i != lowest)
        .map(|&i| (costs[i] - low).powi(2))
        .sum();
    if excess.sqrt() / (peaks.len() - 1) as f32 > options.peak_spread {
        Class::Strong
    } else {
        Class::Weak
    }
}

/// Per pixel, the nearest strong pixel within `radius` (Chebyshev, by a
/// breadth-first search from every strong pixel), `u32::MAX` for none.
pub(crate) fn nearest_strong(classes: &[Class], w: usize, h: usize, radius: usize) -> Vec<u32> {
    let mut nearest = vec![u32::MAX; w * h];
    let mut distance = vec![u16::MAX; w * h];
    let mut queue = VecDeque::new();
    for (i, &c) in classes.iter().enumerate() {
        if c == Class::Strong {
            nearest[i] = i as u32;
            distance[i] = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let d = distance[i];
        if d as usize >= radius {
            continue;
        }
        let (x, y) = ((i % w) as i32, (i / w) as i32);
        for (dx, dy) in [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if distance[j] == u16::MAX {
                distance[j] = d + 1;
                nearest[j] = nearest[i];
                queue.push_back(j);
            }
        }
    }
    nearest
}

/// For weak pixel `(x, y)`, the first strong pixel along each of
/// `options.sectors` rays (radii 2, 4, 8, ... growing by at most 25 pixels
/// a step), or a weak sample's nearest strong pixel if it lies within the
/// sector.
pub(crate) fn sector_anchors(
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    classes: &[Class],
    nearest: &[u32],
    options: &Deformable,
) -> Vec<usize> {
    let sectors = options.sectors.max(1);
    let half = std::f32::consts::PI / sectors as f32;
    let cos_half = half.cos();
    let mut found = Vec::with_capacity(sectors);
    for s in 0..sectors {
        let angle = 2.0 * half * s as f32;
        let (dx, dy) = (angle.cos(), angle.sin());
        let mut radius = 2.0f32;
        loop {
            let qx = (x as f32 + 0.5 + dx * radius).floor();
            let qy = (y as f32 + 0.5 + dy * radius).floor();
            if qx < 0.0 || qy < 0.0 || qx >= w as f32 || qy >= h as f32 {
                break;
            }
            let (qx, qy) = (qx as usize, qy as usize);
            let inside = qx >= BORDER && qy >= BORDER && qx + BORDER < w && qy + BORDER < h;
            if inside {
                let q = qy * w + qx;
                let anchor = if classes[q] == Class::Strong {
                    Some(q)
                } else {
                    (nearest[q] != u32::MAX).then_some(nearest[q] as usize)
                };
                if let Some(a) = anchor {
                    let (ax, ay) = ((a % w) as f32 - x as f32, (a / w) as f32 - y as f32);
                    let norm = (ax * ax + ay * ay).sqrt();
                    if norm > 0.0 && (ax * dx + ay * dy) / norm > cos_half {
                        found.push(a);
                        break;
                    }
                }
            }
            radius = (2.0 * radius).min(radius + 25.0);
        }
    }
    found
}

/// Whether `p` lies in triangle `(a, b, c)` (pixel coordinates), each side
/// longer than 2 pixels.
pub(crate) fn in_triangle(a: [f32; 2], b: [f32; 2], c: [f32; 2], p: [f32; 2]) -> bool {
    let side = |u: [f32; 2], v: [f32; 2]| ((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2)).sqrt();
    if side(a, b) <= 2.0 || side(b, c) <= 2.0 || side(c, a) <= 2.0 {
        return false;
    }
    let cross = |o: [f32; 2], u: [f32; 2], v: [f32; 2]| {
        (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    };
    let area = cross(a, b, c);
    if area.abs() < 1e-6 {
        return false;
    }
    let (d1, d2, d3) = (cross(a, b, p), cross(b, c, p), cross(c, a, p));
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// The plane `(n, w)` with `n · X + w = 0` through three points, `None`
/// when they are collinear.
pub(crate) fn plane_through(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<([f32; 3], f32)> {
    let u = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
    let v = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if norm < 1e-12 {
        return None;
    }
    let n = [n[0] / norm, n[1] / norm, n[2] / norm];
    Some((n, -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2])))
}

/// A plane `(n, w)` with `n · X + w = 0`.
type Plane3 = ([f32; 3], f32);

/// A small deterministic generator for the RANSAC draws.
pub(crate) struct Draws(u32);

impl Draws {
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        // xorshift32
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as usize % n.max(1)
    }
}

/// APD's coplanarity filter: among `anchors` (pixel coordinates and 3D
/// points), the plane through three of them around `center` (pixel and 3D
/// point) with the most anchors within `threshold` of it, ties to the one
/// nearer the centre's point; its inliers (the three first, then by
/// distance), at most [`MAX_ANCHORS`], or `None` with fewer than
/// `min_inliers`.
pub(crate) fn coplanar_anchors(
    center: ([f32; 2], [f32; 3]),
    anchors: &[([f32; 2], [f32; 3])],
    threshold: f32,
    min_inliers: usize,
    draws: &mut Draws,
) -> Option<Vec<usize>> {
    let n = anchors.len();
    if n < 3 {
        return None;
    }
    // Inliers, the centre's distance, the sample and its plane.
    let mut best: Option<(usize, f32, [usize; 3], Plane3)> = None;
    for _ in 0..50 {
        let (i, j, k) = (draws.below(n), draws.below(n), draws.below(n));
        if i == j || j == k || i == k {
            continue;
        }
        if !in_triangle(anchors[i].0, anchors[j].0, anchors[k].0, center.0) {
            continue;
        }
        let Some((normal, w)) = plane_through(anchors[i].1, anchors[j].1, anchors[k].1) else {
            continue;
        };
        let dist = |p: [f32; 3]| (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + w).abs();
        let inliers = anchors.iter().filter(|a| dist(a.1) < threshold).count();
        if inliers < min_inliers {
            continue;
        }
        let center_dist = dist(center.1);
        let better = best.is_none_or(|(count, cd, _, _)| {
            inliers > count || (inliers == count && center_dist < cd)
        });
        if better {
            best = Some((inliers, center_dist, [i, j, k], (normal, w)));
        }
    }
    let (_, _, sample, (normal, w)) = best?;
    let dist = |p: [f32; 3]| (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + w).abs();
    let mut inliers: Vec<(f32, usize)> = anchors
        .iter()
        .enumerate()
        .filter(|(_, a)| dist(a.1) < threshold)
        .map(|(i, a)| (if sample.contains(&i) { -1.0 } else { dist(a.1) }, i))
        .collect();
    inliers.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    Some(
        inliers
            .into_iter()
            .take(MAX_ANCHORS)
            .map(|(_, i)| i)
            .collect(),
    )
}

/// APD's fitted plane: among planes through three of `points` (pixel
/// coordinates and 3D points) whose triangle holds `center`, the one with
/// the least summed distance to the others.
pub(crate) fn fitted_plane(
    center: [f32; 2],
    points: &[([f32; 2], [f32; 3])],
    draws: &mut Draws,
) -> Option<([f32; 3], f32)> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let mut best: Option<(f32, ([f32; 3], f32))> = None;
    for _ in 0..50 {
        let (i, j, k) = (draws.below(n), draws.below(n), draws.below(n));
        if i == j || j == k || i == k {
            continue;
        }
        if !in_triangle(points[i].0, points[j].0, points[k].0, center) {
            continue;
        }
        let Some((normal, w)) = plane_through(points[i].1, points[j].1, points[k].1) else {
            continue;
        };
        let score: f32 = points
            .iter()
            .enumerate()
            .filter(|&(m, _)| m != i && m != j && m != k)
            .map(|(_, p)| (normal[0] * p.1[0] + normal[1] * p.1[1] + normal[2] * p.1[2] + w).abs())
            .sum();
        if best.is_none_or(|(s, _)| score < s) {
            best = Some((score, (normal, w)));
            if score == 0.0 {
                break;
            }
        }
    }
    best.map(|(_, plane)| plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_sharp_minimum_near_the_depth_is_strong() {
        let options = Deformable::default();
        let mut costs = vec![0.8f32; 61];
        costs[30] = 0.1;
        assert_eq!(classify(&costs, &options), Class::Strong);
        // Too costly a minimum, or too far from the depth.
        costs[30] = 0.3;
        assert_eq!(classify(&costs, &options), Class::Weak);
        costs[30] = 0.8;
        costs[40] = 0.1;
        assert_eq!(classify(&costs, &options), Class::Weak);
        // Flat: no minimum.
        assert_eq!(classify(&[0.4; 61], &options), Class::Weak);
        // Two minima: strong only if the other is clearly worse.
        let mut two = vec![0.8f32; 61];
        two[30] = 0.1;
        two[50] = 0.2;
        assert_eq!(classify(&two, &options), Class::Weak);
        two[50] = 0.5;
        assert_eq!(classify(&two, &options), Class::Strong);
    }

    #[test]
    fn the_nearest_strong_pixel_spreads_within_the_radius() {
        let (w, h) = (20, 10);
        let mut classes = vec![Class::Weak; w * h];
        classes[5 * w + 3] = Class::Strong;
        let nearest = nearest_strong(&classes, w, h, 4);
        assert_eq!(nearest[5 * w + 6], (5 * w + 3) as u32);
        assert_eq!(nearest[w + 3], (5 * w + 3) as u32);
        assert_eq!(nearest[5 * w + 8], u32::MAX);
    }

    #[test]
    fn anchors_come_from_every_side_and_must_be_coplanar() {
        // A weak disc in a strong frame: every sector finds the frame.
        let (w, h) = (101, 101);
        let mut classes = vec![Class::Strong; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - 50.0, y as f32 - 50.0);
                if dx * dx + dy * dy < 30.0 * 30.0 {
                    classes[y * w + x] = Class::Weak;
                }
            }
        }
        let nearest = nearest_strong(&classes, w, h, 100);
        let options = Deformable::default();
        let found = sector_anchors((50, 50), (w, h), &classes, &nearest, &options);
        assert_eq!(found.len(), options.sectors);
        // On the plane z = 4 (pixel = point), plus one off it.
        let point = |a: usize| {
            let (x, y) = ((a % w) as f32, (a / w) as f32);
            ([x, y], [x / 100.0, y / 100.0, 4.0])
        };
        let mut anchors: Vec<_> = found.iter().map(|&a| point(a)).collect();
        anchors[3].1[2] = 4.5;
        let center = ([50.0, 50.0], [0.5, 0.5, 4.0]);
        let kept = coplanar_anchors(center, &anchors, 0.01, 6, &mut Draws::new(7)).unwrap();
        assert_eq!(kept.len(), MAX_ANCHORS);
        assert!(!kept.contains(&3));
        let (n, d) = fitted_plane(center.0, &anchors, &mut Draws::new(7)).unwrap();
        assert!((n[2].abs() - 1.0).abs() < 1e-4 && (d.abs() - 4.0).abs() < 1e-3);
    }
}
