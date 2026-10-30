//! Training configuration and the host-side helpers shared by the trainers
//! (the wgpu `trainer` module and the CUDA one in `visloc-gsplat-cuda`).
//! Needs no GPU feature.

use visloc_gsplat_core::gaussian::{Gaussian, Scene};

use crate::densify::{BrushRefineConfig, DensifyConfig, Population};

/// Training hyper-parameters.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub steps: usize,
    /// Mean learning rate at the start / end, times the scene extent.
    pub lr_mean: f32,
    pub lr_mean_final: f32,
    pub lr_quat: f32,
    pub lr_scale: f32,
    pub lr_opacity: f32,
    pub lr_sh_dc: f32,
    pub lr_sh_rest: f32,
    pub background: [f32; 3],
    pub seed: u64,
    /// Weight of the D-SSIM term (0 = pure L1).
    pub ssim_weight: f32,
    /// `None` keeps the initial gaussians fixed in number.
    pub densify: Option<DensifyConfig>,
    /// brush's refine strategy instead of the Inria rule (`densify` is then
    /// ignored); see [`BrushRefineConfig`]. In this mode `lr_mean` scales
    /// with the splats' bound size (brush) rather than the camera extent.
    pub brush_refine: Option<BrushRefineConfig>,
    /// Exponential decay target of the scale learning rate (`None` keeps
    /// `lr_scale` constant, the Inria schedule).
    pub lr_scale_final: Option<f32>,
    /// Raise the evaluated SH degree by one every this many steps (Inria /
    /// brush schedule; early steps skip the higher bands). 0 evaluates the
    /// scene's full degree from the start.
    pub sh_degree_interval: usize,
    /// Weight of the depth-normal consistency loss (2DGS / PGSR style) on
    /// the renderer's geometry channels; 0 = off. Active from
    /// `normal_start` on.
    pub normal_weight: f32,
    pub normal_start: usize,
    /// brush's auxiliary losses (see `shaders/aux_loss.wgsl`): opacity-logit
    /// and scale weights, applied with weight `aux_loss_time - t` (t = step /
    /// steps, clamped to [0, 1]). 0 = off.
    pub opac_loss_weight: f32,
    pub scale_loss_weight: f32,
    pub aux_loss_time: f32,
    /// Per-training-image affine colour transform (3x4) applied to the
    /// render before the loss (see `shaders/appearance.wgsl`); held-out
    /// views render without it.
    pub appearance: bool,
    pub appearance_lr: f32,
    /// Pull of the transforms towards identity.
    pub appearance_reg: f32,
    /// Leave out of the loss the pixels whose image alpha is below 128
    /// (masked-out content, e.g. people and cars a dataset blacked out).
    /// Images without alpha train every pixel. Off: alpha is ignored.
    pub alpha_mask: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 7000,
            lr_mean: 1.6e-4,
            lr_mean_final: 1.6e-6,
            lr_quat: 1e-3,
            lr_scale: 5e-3,
            lr_opacity: 5e-2,
            lr_sh_dc: 2.5e-3,
            lr_sh_rest: 2.5e-3 / 20.0,
            background: [0.0; 3],
            seed: 42,
            ssim_weight: 0.2,
            densify: Some(DensifyConfig::default()),
            sh_degree_interval: 1000,
            brush_refine: None,
            lr_scale_final: None,
            normal_weight: 0.0,
            normal_start: 7000,
            opac_loss_weight: 0.0,
            scale_loss_weight: 0.0,
            aux_loss_time: 0.9,
            appearance: false,
            appearance_lr: 2e-3,
            appearance_reg: 1e-2,
            alpha_mask: false,
        }
    }
}

impl TrainConfig {
    /// brush 0.3's defaults: its refine strategy, mean noise and learning
    /// rates (mean 2e-5 -> 1e-6 times the bound size, scale 1e-2 -> 6e-3,
    /// opacity 1e-2, SH DC 2e-3 with the rest / 20).
    pub fn brush_preset() -> Self {
        Self {
            lr_mean: 2e-5,
            lr_mean_final: 1e-6,
            lr_quat: 1e-3,
            lr_scale: 1e-2,
            lr_scale_final: Some(6e-3),
            lr_opacity: 1e-2,
            lr_sh_dc: 2e-3,
            lr_sh_rest: 2e-3 / 20.0,
            densify: None,
            brush_refine: Some(BrushRefineConfig::default()),
            opac_loss_weight: 1e-9,
            scale_loss_weight: 1e-8,
            ..Self::default()
        }
    }
}

/// Pack RGB as RGBA8 (R in the low byte); alpha 0 marks a pixel the loss
/// leaves out (`valid` false), 255 one it trains.
pub fn pack_rgba8(rgb: &[[f32; 3]], valid: Option<&[bool]>) -> Vec<u32> {
    rgb.iter()
        .enumerate()
        .map(|(i, p)| {
            let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            let a = if valid.is_none_or(|v| v[i]) { 255 } else { 0 };
            c(p[0]) | (c(p[1]) << 8) | (c(p[2]) << 16) | (a << 24)
        })
        .collect()
}

/// brush's bounds: the per-axis central `percentile` box of `points`;
/// returns (centre, median axis length).
pub fn central_bounds(points: &[[f32; 3]], percentile: f32) -> Option<([f32; 3], f32)> {
    if points.is_empty() {
        return None;
    }
    let mut center = [0.0f32; 3];
    let mut sizes = [0.0f32; 3];
    for k in 0..3 {
        let mut v: Vec<f32> = points
            .iter()
            .map(|p| p[k])
            .filter(|x| x.is_finite())
            .collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        let lo = ((1.0 - percentile) / 2.0 * v.len() as f32) as usize;
        let hi = (((1.0 + percentile) / 2.0 * v.len() as f32) as usize).min(v.len() - 1);
        center[k] = 0.5 * (v[lo] + v[hi]);
        sizes[k] = v[hi] - v[lo];
    }
    sizes.sort_by(|a, b| a.total_cmp(b));
    Some((center, sizes[1].max(1e-6)))
}

pub fn central_bound_size(points: &[[f32; 3]], percentile: f32) -> Option<f32> {
    central_bounds(points, percentile).map(|(_, s)| s)
}

/// Gaussians from a population (the device parameter layouts).
pub fn population_to_scene(pop: &Population, degree: u32) -> Scene {
    let cpc2 = ((degree + 1) * (degree + 1)) as usize;
    let (t, o, sh) = (&pop.transforms.values, &pop.opacity.values, &pop.sh.values);
    let gaussians = (0..o.len())
        .map(|i| {
            let r = &t[i * 10..i * 10 + 10];
            let s = &sh[i * 3 * cpc2..(i + 1) * 3 * cpc2];
            Gaussian {
                mean: nalgebra::Vector3::new(r[0], r[1], r[2]),
                rotation: nalgebra::Quaternion::new(r[3], r[4], r[5], r[6]),
                scale_log: nalgebra::Vector3::new(r[7], r[8], r[9]),
                opacity_logit: o[i],
                sh_dc: [s[0], s[1], s[2]],
                sh_rest: s[3..].to_vec(),
                sh_degree: degree,
            }
        })
        .collect();
    Scene::new(gaussians, degree)
}
