//! Initial gaussians from SfM points.
//!
//! One gaussian per point, as in the Inria recipe: colour from the point's RGB
//! (as the SH DC term), isotropic scale from the mean distance to its three
//! nearest neighbours, identity rotation, opacity 0.1, higher SH zero.

use std::path::Path;

use nalgebra::{Quaternion, Vector3};
use visloc_gsplat_core::gaussian::{sh_rest_coeffs_per_channel, Gaussian, Scene};
use visloc_gsplat_core::sh::SH_C0;

/// A coloured SfM point.
#[derive(Debug, Clone, Copy)]
pub struct ColoredPoint {
    pub position: Vector3<f32>,
    pub rgb: [u8; 3],
}

/// Errors reading `points3D.txt`.
#[derive(Debug, thiserror::Error)]
pub enum PointsError {
    #[error("reading {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("{path}:{line}: malformed point line")]
    Malformed {
        path: std::path::PathBuf,
        line: usize,
    },
}

/// Read a COLMAP text `points3D.txt` (`ID X Y Z R G B ERROR TRACK[]`).
pub fn read_points3d_txt(path: impl AsRef<Path>) -> Result<Vec<ColoredPoint>, PointsError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| PointsError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().take(7).collect();
        let bad = || PointsError::Malformed {
            path: path.to_path_buf(),
            line: i + 1,
        };
        if f.len() < 7 {
            return Err(bad());
        }
        let num = |k: usize| f[k].parse::<f32>().map_err(|_| bad());
        let byte = |k: usize| f[k].parse::<u8>().map_err(|_| bad());
        out.push(ColoredPoint {
            position: Vector3::new(num(1)?, num(2)?, num(3)?),
            rgb: [byte(4)?, byte(5)?, byte(6)?],
        });
    }
    Ok(out)
}

/// Mean distance from each point to its `k` nearest other points, via a
/// uniform hash grid (cell ~ the average spacing). Points with fewer than `k`
/// neighbours in the searched rings get the mean over what was found; an
/// isolated point gets the global average.
pub fn knn_mean_distance(points: &[Vector3<f32>], k: usize) -> Vec<f32> {
    let n = points.len();
    if n < 2 || k == 0 {
        return vec![0.01; n];
    }
    // Exact k-NN with a k-d tree. (A uniform grid sized from the bounding
    // box degenerates to O(n^2) when a few far background points blow up
    // the box, as in Mip-NeRF 360 scenes.)
    let tree = KdTree::build(points);
    let mut out = vec![f32::NAN; n];
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, p) in points.iter().enumerate() {
        heap.clear();
        tree.knn(points, p, i as u32, k, &mut heap);
        if !heap.is_empty() {
            out[i] = heap.iter().map(|(d2, _)| d2.sqrt()).sum::<f32>() / heap.len() as f32;
        }
    }
    let (sum, cnt) = out
        .iter()
        .filter(|d| d.is_finite() && **d > 0.0)
        .fold((0.0f64, 0usize), |(s, c), d| (s + *d as f64, c + 1));
    let fallback = if cnt > 0 {
        (sum / cnt as f64) as f32
    } else {
        0.01
    };
    for d in &mut out {
        if !d.is_finite() || *d <= 0.0 {
            *d = fallback;
        }
    }
    out
}

/// Implicit k-d tree over point indices (median splits, cycling axes).
struct KdTree {
    /// Permuted point indices; node `[lo, hi)` splits at `mid = (lo + hi) / 2`.
    idx: Vec<u32>,
}

impl KdTree {
    fn build(points: &[Vector3<f32>]) -> Self {
        let mut idx: Vec<u32> = (0..points.len() as u32).collect();
        fn rec(points: &[Vector3<f32>], idx: &mut [u32], depth: usize) {
            if idx.len() <= 1 {
                return;
            }
            let axis = depth % 3;
            let mid = idx.len() / 2;
            idx.select_nth_unstable_by(mid, |a, b| {
                points[*a as usize][axis].total_cmp(&points[*b as usize][axis])
            });
            let (left, right) = idx.split_at_mut(mid);
            rec(points, left, depth + 1);
            rec(points, &mut right[1..], depth + 1);
        }
        rec(points, &mut idx, 0);
        Self { idx }
    }

    /// The `k` nearest neighbours of `q` (excluding index `skip`) as
    /// `(squared distance, index)`, in `heap` (unordered).
    fn knn(
        &self,
        points: &[Vector3<f32>],
        q: &Vector3<f32>,
        skip: u32,
        k: usize,
        heap: &mut Vec<(f32, u32)>,
    ) {
        fn worst(heap: &[(f32, u32)]) -> (usize, f32) {
            heap.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |acc, (i, (d, _))| {
                    if *d > acc.1 {
                        (i, *d)
                    } else {
                        acc
                    }
                })
        }
        #[allow(clippy::too_many_arguments)]
        fn rec(
            idx: &[u32],
            points: &[Vector3<f32>],
            q: &Vector3<f32>,
            skip: u32,
            k: usize,
            depth: usize,
            heap: &mut Vec<(f32, u32)>,
        ) {
            if idx.is_empty() {
                return;
            }
            let axis = depth % 3;
            let mid = idx.len() / 2;
            let j = idx[mid];
            if j != skip {
                let d2 = (points[j as usize] - q).norm_squared();
                if heap.len() < k {
                    heap.push((d2, j));
                } else {
                    let (w, wd) = worst(heap);
                    if d2 < wd {
                        heap[w] = (d2, j);
                    }
                }
            }
            let diff = q[axis] - points[j as usize][axis];
            let (near, far) = if diff < 0.0 {
                (&idx[..mid], &idx[mid + 1..])
            } else {
                (&idx[mid + 1..], &idx[..mid])
            };
            rec(near, points, q, skip, k, depth + 1, heap);
            if heap.len() < k || diff * diff < worst(heap).1 {
                rec(far, points, q, skip, k, depth + 1, heap);
            }
        }
        rec(&self.idx, points, q, skip, k, 0, heap);
    }
}

/// Seed a scene of SH degree `sh_degree` from coloured points.
pub fn seed_scene(points: &[ColoredPoint], sh_degree: u32) -> Scene {
    let positions: Vec<Vector3<f32>> = points.iter().map(|p| p.position).collect();
    let dist = knn_mean_distance(&positions, 3);
    let rest = 3 * sh_rest_coeffs_per_channel(sh_degree);
    // logit(0.1)
    let opacity_logit = (0.1f32 / 0.9).ln();
    let gaussians = points
        .iter()
        .zip(dist)
        .map(|(p, d)| {
            let ls = d.max(1e-7).ln();
            Gaussian {
                mean: p.position,
                scale_log: Vector3::new(ls, ls, ls),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                opacity_logit,
                sh_dc: p.rgb.map(|c| (c as f32 / 255.0 - 0.5) / SH_C0),
                sh_rest: vec![0.0; rest],
                sh_degree,
            }
        })
        .collect();
    Scene::new(gaussians, sh_degree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_matches_brute_force_with_far_outliers() {
        // Dense cluster plus a few far points (the case that made the old
        // bounding-box grid quadratic).
        let mut state = 12345u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut pts: Vec<Vector3<f32>> = (0..2000)
            .map(|_| Vector3::new(rnd(), rnd(), rnd()))
            .collect();
        pts.extend((0..20).map(|i| Vector3::new(500.0 + i as f32, -300.0, 900.0 * rnd())));
        let got = knn_mean_distance(&pts, 3);
        for (i, p) in pts.iter().enumerate() {
            let mut d: Vec<f32> = pts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, q)| (q - p).norm())
                .collect();
            d.sort_by(|a, b| a.total_cmp(b));
            let want = (d[0] + d[1] + d[2]) / 3.0;
            assert!(
                (got[i] - want).abs() <= 1e-4 * want.max(1.0),
                "point {i}: {} vs {want}",
                got[i]
            );
        }
    }

    #[test]
    fn knn_on_a_lattice() {
        // Unit lattice: the 3 nearest neighbours of an interior point are at 1.
        let mut pts = Vec::new();
        for x in 0..6 {
            for y in 0..6 {
                for z in 0..6 {
                    pts.push(Vector3::new(x as f32, y as f32, z as f32));
                }
            }
        }
        let d = knn_mean_distance(&pts, 3);
        let interior = pts
            .iter()
            .position(|p| *p == Vector3::new(2.0, 3.0, 2.0))
            .unwrap();
        assert!((d[interior] - 1.0).abs() < 1e-5, "got {}", d[interior]);
        assert!(d.iter().all(|x| x.is_finite() && *x > 0.0));
    }

    #[test]
    fn seed_colours_round_trip_through_dc() {
        let scene = seed_scene(
            &[
                ColoredPoint {
                    position: Vector3::new(0.0, 0.0, 0.0),
                    rgb: [255, 128, 0],
                },
                ColoredPoint {
                    position: Vector3::new(1.0, 0.0, 0.0),
                    rgb: [0, 0, 0],
                },
            ],
            3,
        );
        let g = &scene.gaussians[0];
        let c = g.sh_dc.map(|d| d * SH_C0 + 0.5);
        assert!((c[0] - 1.0).abs() < 1e-5 && (c[2] - 0.0).abs() < 1e-5);
        assert_eq!(g.sh_rest.len(), 45);
        assert!(g.sh_is_consistent());
    }
}
