//! The trained parameters, structure of arrays.

use nalgebra::{Quaternion, Vector3};
use visloc_gsplat_core::gaussian::{Gaussian, Scene};

/// Geometry parameters per gaussian: mean (3), log-scale (3), quaternion
/// `w x y z` (4, unnormalised), opacity logit (1).
pub const NGEO: usize = 11;
/// SH coefficients per channel at degree 3 (DC first).
pub const NSH: usize = 16;
/// Colour floats per gaussian (`[channel][coefficient]`).
pub const NCOL: usize = 3 * NSH;

#[derive(Clone, Debug)]
pub struct Model {
    pub geo: Vec<[f32; NGEO]>,
    /// `NCOL` floats per gaussian, `col[g * NCOL + ch * NSH + k]`.
    pub col: Vec<f32>,
    /// SH degree the colour is evaluated at (coefficients past it are zero
    /// in the basis and get no gradient).
    pub active_degree: u32,
    /// Screen-space blur added to every 2D covariance, px² (0.3, as the
    /// Inria rasterizer; strided training widens it, see `train`).
    pub blur: f32,
}

impl Model {
    pub fn len(&self) -> usize {
        self.geo.len()
    }

    pub fn is_empty(&self) -> bool {
        self.geo.is_empty()
    }

    pub fn color(&self, id: usize) -> &[f32] {
        &self.col[id * NCOL..(id + 1) * NCOL]
    }

    pub fn from_scene(scene: &Scene) -> Self {
        let mut geo = Vec::with_capacity(scene.gaussians.len());
        let mut col = vec![0.0f32; scene.gaussians.len() * NCOL];
        for (i, g) in scene.gaussians.iter().enumerate() {
            geo.push([
                g.mean.x,
                g.mean.y,
                g.mean.z,
                g.scale_log.x,
                g.scale_log.y,
                g.scale_log.z,
                g.rotation.w,
                g.rotation.i,
                g.rotation.j,
                g.rotation.k,
                g.opacity_logit,
            ]);
            let per = g.sh_rest.len() / 3;
            for ch in 0..3 {
                col[i * NCOL + ch * NSH] = g.sh_dc[ch];
                for k in 0..per.min(NSH - 1) {
                    col[i * NCOL + ch * NSH + 1 + k] = g.sh_rest[ch * per + k];
                }
            }
        }
        Self {
            geo,
            col,
            active_degree: 0,
            blur: 0.3,
        }
    }

    /// The scene at SH degree 3 (Inria PLY layout).
    pub fn to_scene(&self) -> Scene {
        let gaussians = self
            .geo
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let c = self.color(i);
                let mut rest = vec![0.0f32; 3 * (NSH - 1)];
                for ch in 0..3 {
                    for k in 1..NSH {
                        rest[ch * (NSH - 1) + k - 1] = c[ch * NSH + k];
                    }
                }
                Gaussian {
                    mean: Vector3::new(g[0], g[1], g[2]),
                    scale_log: Vector3::new(g[3], g[4], g[5]),
                    rotation: Quaternion::new(g[6], g[7], g[8], g[9]),
                    opacity_logit: g[10],
                    sh_dc: [c[0], c[NSH], c[2 * NSH]],
                    sh_rest: rest,
                    sh_degree: 3,
                }
            })
            .collect();
        Scene::new(gaussians, 3)
    }
}
