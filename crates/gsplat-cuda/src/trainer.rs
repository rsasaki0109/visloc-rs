//! CUDA trainer: the wgpu trainer's loop (`visloc_gsplat_train::trainer`)
//! on the native training step, for brush's refine strategy
//! ([`TrainConfig::brush_preset`]) or a fixed population.
//!
//! The schedule is the wgpu trainer's: views reshuffled every epoch, the
//! mean's learning rate decaying from `lr_mean` to `lr_mean_final` (times
//! the splats' bound size), the scale's from `lr_scale` to `lr_scale_final`,
//! the SH degree raised every `sh_degree_interval` steps, brush's aux losses
//! fading out at `aux_loss_time`, and every `refine_every` steps a refine
//! whose actions the host picks from the device's statistics. The
//! depth-normal loss (`normal_weight`) runs from `normal_start` on, as in
//! the wgpu trainer. Not supported yet: the Inria densification rule and
//! per-image appearance.

use std::fmt;

use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::config::{
    central_bound_size, central_bounds, pack_rgba8, population_to_scene, TrainConfig,
};
use visloc_gsplat_train::dataset::{
    load_view_rgb, load_view_rgb_mask, Dataset, DatasetError, View,
};
use visloc_gsplat_train::densify::{BrushRefineConfig, DensifyReport, Group, Population};

use crate::{CudaGsplat, GscCamera, GsplatCudaError, StepParams};

#[derive(Debug)]
pub enum CudaTrainError {
    Cuda(GsplatCudaError),
    Dataset(DatasetError),
    NoViews,
    MixedResolution(u32, u32, u32, u32),
    Unsupported(&'static str),
}

impl fmt::Display for CudaTrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => e.fmt(f),
            Self::Dataset(e) => e.fmt(f),
            Self::NoViews => write!(f, "the dataset has no training views"),
            Self::MixedResolution(w, h, w2, h2) => {
                write!(f, "views of different sizes: {w}x{h} and {w2}x{h2}")
            }
            Self::Unsupported(what) => write!(f, "not supported by the CUDA trainer yet: {what}"),
        }
    }
}

impl std::error::Error for CudaTrainError {}

impl From<GsplatCudaError> for CudaTrainError {
    fn from(e: GsplatCudaError) -> Self {
        Self::Cuda(e)
    }
}

impl From<DatasetError> for CudaTrainError {
    fn from(e: DatasetError) -> Self {
        Self::Dataset(e)
    }
}

pub struct CudaTrainer {
    gs: CudaGsplat,
    cfg: TrainConfig,
    views: Vec<View>,
    order: Vec<usize>,
    rng: u64,
    step: usize,
    sh_degree: u32,
    active_sh: u32,
    extent: f32,
    bound_size: f32,
    last_refine: Option<DensifyReport>,
    sparse_sh_adam: bool,
    depth_weight: f32,
    last_depth_loss: f32,
    last_normal_loss: f32,
}

impl fmt::Debug for CudaTrainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaTrainer")
            .field("step", &self.step)
            .field("views", &self.views.len())
            .finish_non_exhaustive()
    }
}

impl CudaTrainer {
    /// Upload `init` and the dataset's training images.
    pub fn new(
        mut gs: CudaGsplat,
        dataset: &Dataset,
        init: &Scene,
        cfg: TrainConfig,
    ) -> Result<Self, CudaTrainError> {
        if cfg.brush_refine.is_none() && cfg.densify.is_some() {
            return Err(CudaTrainError::Unsupported(
                "the Inria densification rule (use the brush strategy)",
            ));
        }
        if cfg.appearance {
            return Err(CudaTrainError::Unsupported("per-image appearance"));
        }
        let views = dataset.train.clone();
        let first = views.first().ok_or(CudaTrainError::NoViews)?;
        let (width, height) = (first.camera.camera.width, first.camera.camera.height);
        for v in dataset.train.iter().chain(&dataset.eval) {
            let (w, h) = (v.camera.camera.width, v.camera.camera.height);
            if (w, h) != (width, height) {
                return Err(CudaTrainError::MixedResolution(width, height, w, h));
            }
        }
        // Scene extent: 1.1x the largest camera distance from their centroid.
        let centers: Vec<_> = views.iter().map(|v| v.camera.camera_center()).collect();
        let centroid = centers
            .iter()
            .fold(nalgebra::Vector3::zeros(), |a, c| a + c)
            / centers.len() as f32;
        let extent = 1.1
            * centers
                .iter()
                .map(|c| (c - centroid).norm())
                .fold(0.0f32, f32::max)
                .max(1e-3);

        let mut images = Vec::with_capacity(views.len() * (width * height) as usize);
        for v in &views {
            if cfg.alpha_mask {
                let (rgb, valid) = load_view_rgb_mask(v)?;
                images.extend(pack_rgba8(&rgb, valid.as_deref()));
            } else {
                images.extend(pack_rgba8(&load_view_rgb(v)?, None));
            }
        }
        gs.set_images(width, height, &images)?;
        gs.set_scene(&PackedScene::from_scene(init))?;

        let init_means: Vec<[f32; 3]> = init
            .gaussians
            .iter()
            .map(|g| [g.mean.x, g.mean.y, g.mean.z])
            .collect();
        let bound_size = central_bound_size(&init_means, 0.75).unwrap_or(extent);
        let order = (0..views.len()).collect();
        Ok(Self {
            gs,
            rng: cfg.seed.max(1),
            cfg,
            views,
            order,
            step: 0,
            sh_degree: init.sh_degree,
            active_sh: init.sh_degree,
            extent,
            bound_size,
            last_refine: None,
            sparse_sh_adam: false,
            depth_weight: 0.0,
            last_depth_loss: 0.0,
            last_normal_loss: 0.0,
        })
    }

    /// Run the SH Adam only on the gaussians each step blends (off by
    /// default: the wgpu trainer's dense Adam). The other groups stay dense,
    /// so brush's aux losses still fade the gaussians no view sees.
    pub fn set_sparse_sh_adam(&mut self, on: bool) {
        self.sparse_sh_adam = on;
    }

    pub fn steps_done(&self) -> usize {
        self.step
    }

    pub fn num_gaussians(&self) -> usize {
        self.gs.num_gaussians()
    }

    pub fn extent(&self) -> f32 {
        self.extent
    }

    /// The last refine's counts (once per refine).
    pub fn take_refine_report(&mut self) -> Option<DensifyReport> {
        self.last_refine.take()
    }

    /// GPU milliseconds per stage of the last step.
    pub fn stage_times(&self) -> Vec<(String, f32)> {
        self.gs.stage_times()
    }

    fn next_view(&mut self) -> usize {
        // Reshuffle every epoch (xorshift, deterministic from the seed).
        let k = self.step % self.order.len();
        if k == 0 {
            for i in (1..self.order.len()).rev() {
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 7;
                self.rng ^= self.rng << 17;
                let j = (self.rng % (i as u64 + 1)) as usize;
                self.order.swap(i, j);
            }
        }
        self.order[k]
    }

    /// One optimisation step on the next training view.
    pub fn step(&mut self) -> Result<(), CudaTrainError> {
        let vi = self.next_view();
        let brush = self.cfg.brush_refine.clone();
        let frac = (self.step as f32 / self.cfg.steps.max(1) as f32).min(1.0);
        let lr_mean = (self.cfg.lr_mean.ln() * (1.0 - frac) + self.cfg.lr_mean_final.ln() * frac)
            .exp()
            * if brush.is_some() {
                self.bound_size
            } else {
                self.extent
            };
        let lr_scale = match self.cfg.lr_scale_final {
            Some(end) => (self.cfg.lr_scale.ln() * (1.0 - frac) + end.ln() * frac).exp(),
            None => self.cfg.lr_scale,
        };
        self.active_sh = if self.cfg.sh_degree_interval > 0 {
            ((self.step / self.cfg.sh_degree_interval) as u32).min(self.sh_degree)
        } else {
            self.sh_degree
        };
        let aux_w = (self.cfg.aux_loss_time - frac).clamp(0.0, 1.0);
        let t = (self.step + 1) as f32;
        let (beta1, beta2) = (0.9f32, 0.999f32);
        let (noise_scale, noise_max) = match &brush {
            Some(b) => (lr_mean * b.mean_noise_weight, 0.25 * self.bound_size),
            None => (0.0, 0.0),
        };
        let params = StepParams {
            camera: GscCamera::from_view(&self.views[vi].camera),
            image: vi as u32,
            sh_active_degree: self.active_sh as i32,
            background: self.cfg.background,
            ssim_weight: self.cfg.ssim_weight,
            lr: [
                lr_mean,
                self.cfg.lr_quat,
                lr_scale,
                self.cfg.lr_opacity,
                self.cfg.lr_sh_dc,
                self.cfg.lr_sh_rest,
            ],
            beta1,
            beta2,
            eps: 1e-15,
            bc1: 1.0 - beta1.powf(t),
            bc2: 1.0 - beta2.powf(t),
            scale_aux_coef: self.cfg.scale_loss_weight * aux_w / self.bound_size.max(1e-6),
            opacity_aux_coef: self.cfg.opac_loss_weight * aux_w,
            // brush gathers refine statistics for the whole run.
            accumulate_stats: brush.is_some() as u32,
            noise_seed: (self.cfg.seed as u32) ^ (self.step as u32).wrapping_mul(0x85EB_CA6B),
            noise_scale,
            noise_max,
            sparse_sh_adam: self.sparse_sh_adam as u32,
            depth_weight: self.depth_weight,
            normal_weight: if self.step >= self.cfg.normal_start {
                self.cfg.normal_weight
            } else {
                0.0
            },
        };
        self.gs.train_step(&params)?;
        self.step += 1;
        if let Some(b) = brush {
            if self.step.is_multiple_of(b.refine_every.max(1)) {
                self.brush_refine(&b)?;
            }
        }
        Ok(())
    }

    /// brush's refine, as `Trainer::brush_refine_now`: prune nearly
    /// transparent, degenerate and far-away gaussians; replace them by
    /// splitting as many sampled by opacity; until `growth_stop`, split a
    /// further fraction of those above the refine threshold, sampled by
    /// their weight.
    fn brush_refine(&mut self, b: &BrushRefineConfig) -> Result<(), CudaTrainError> {
        let n = self.gs.num_gaussians();
        if n == 0 {
            return Ok(());
        }
        let scene = self.gs.scene()?;
        let stats = self.gs.refine_stats()?;
        let transforms = &scene.transforms;
        let means: Vec<[f32; 3]> = transforms
            .chunks_exact(10)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        let (center, size) = central_bounds(&means, 0.75).unwrap_or(([0.0; 3], self.bound_size));
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let opacity: Vec<f32> = scene.opacity.iter().map(|&x| sigmoid(x)).collect();
        let pruned: Vec<bool> = (0..n)
            .map(|i| {
                let t = &transforms[i * 10..i * 10 + 10];
                opacity[i] < b.min_opacity
                    || t[7..10].iter().any(|&ls| ls < -15.0)
                    || (0..3).any(|k| (t[k] - center[k]).abs() > size * 10.0)
            })
            .collect();
        let pruned_count = pruned.iter().filter(|p| **p).count();

        let mut add = vec![false; n];
        let rng = &mut self.rng;
        // Weighted sampling without replacement (Efraimidis-Spirakis: the
        // `count` smallest keys -ln(u) / w), as the wgpu trainer.
        let mut sample = |weights: &[f32], count: usize, add: &mut Vec<bool>| {
            let mut keys: Vec<(f64, usize)> = Vec::with_capacity(weights.len());
            for (j, &w) in weights.iter().enumerate() {
                if w > 0.0 && w.is_finite() {
                    *rng ^= *rng << 13;
                    *rng ^= *rng >> 7;
                    *rng ^= *rng << 17;
                    let u = ((*rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                    keys.push((-u.ln() / w as f64, j));
                }
            }
            let count = count.min(keys.len());
            if count == 0 {
                return;
            }
            if count < keys.len() {
                keys.select_nth_unstable_by(count - 1, |a, b| a.0.total_cmp(&b.0));
            }
            for &(_, j) in &keys[..count] {
                add[j] = true;
            }
        };
        if pruned_count > 0 {
            let w: Vec<f32> = (0..n)
                .map(|i| if pruned[i] { 0.0 } else { opacity[i] })
                .collect();
            sample(&w, pruned_count, &mut add);
        }
        let mut grown = 0;
        if self.step < b.growth_stop {
            let above: Vec<bool> = (0..n)
                .map(|i| {
                    !pruned[i]
                        && stats.weight_max[i] / stats.seen[i].max(1.0) > b.growth_grad_threshold
                })
                .collect();
            let threshold_count = above.iter().filter(|a| **a).count();
            let grow = ((threshold_count as f32 * b.growth_select_fraction).round() as usize)
                .saturating_sub(pruned_count);
            let current = n - pruned_count + add.iter().filter(|a| **a).count();
            let grow = grow.min(b.max_gaussians.saturating_sub(current));
            if grow > 0 {
                let w: Vec<f32> = (0..n)
                    .map(|i| if above[i] { stats.weight_max[i] } else { 0.0 })
                    .collect();
                let before = add.iter().filter(|a| **a).count();
                sample(&w, grow, &mut add);
                grown = add.iter().filter(|a| **a).count() - before;
            }
        }
        let actions: Vec<u32> = (0..n)
            .map(|i| {
                if pruned[i] {
                    0
                } else if add[i] {
                    4
                } else {
                    1
                }
            })
            .collect();
        let seed = (self.cfg.seed as u32) ^ (self.step as u32).wrapping_mul(0x9E37_79B9);
        let n_after = self.gs.apply_actions(&actions, seed)?;
        self.bound_size = size;
        self.last_refine = Some(DensifyReport {
            cloned: grown,
            split: actions.iter().filter(|a| **a == 4).count() - grown,
            pruned: pruned_count,
            before: n,
            after: n_after,
        });
        Ok(())
    }

    /// Mean L1 loss over the steps since the last call (and the mean
    /// depth-prior and depth-normal losses, kept for
    /// [`last_depth_loss`](Self::last_depth_loss) and
    /// [`last_normal_loss`](Self::last_normal_loss)).
    pub fn take_mean_loss(&mut self) -> Result<f32, CudaTrainError> {
        let sums = self.gs.take_loss()?;
        let steps = sums.steps.max(1) as f64;
        self.last_depth_loss = (sums.depth / steps) as f32;
        self.last_normal_loss = (sums.normal / steps) as f32;
        Ok((sums.l1 / steps) as f32)
    }

    /// Mean depth-prior loss `|D - P| / P` (averaged over all pixels) of the
    /// steps before the last [`take_mean_loss`](Self::take_mean_loss).
    pub fn last_depth_loss(&self) -> f32 {
        self.last_depth_loss
    }

    /// Mean depth-normal loss `1 - N . n_d` (averaged over all pixels) of the
    /// steps before the last [`take_mean_loss`](Self::take_mean_loss); steps
    /// before `normal_start` count as 0.
    pub fn last_normal_loss(&self) -> f32 {
        self.last_normal_loss
    }

    /// Load a depth prior per training view from `dir/<image name>.bin` (as
    /// written by `visloc-mvs --save-fused-depth`: width and height `u32`,
    /// then camera-z `f32` depths, 0 where there is none), and train them
    /// with `weight`. A map of another size (e.g. depth estimated at half the
    /// resolution) is resampled to the view's by nearest neighbour; a view
    /// without a readable file gets no prior. Returns the number of views with
    /// one.
    pub fn load_depth_priors(
        &mut self,
        dir: &std::path::Path,
        weight: f32,
    ) -> Result<usize, CudaTrainError> {
        let first = &self.views[0].camera.camera;
        let (w, h) = (first.width, first.height);
        let per = (w * h) as usize;
        let mut depths = vec![0.0f32; per * self.views.len()];
        let mut found = 0;
        for (k, view) in self.views.iter().enumerate() {
            let Ok(bytes) = std::fs::read(dir.join(format!("{}.bin", view.name))) else {
                continue;
            };
            if bytes.len() < 8 {
                continue;
            }
            let pw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let ph = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if pw == 0 || ph == 0 || bytes.len() != 8 + (pw * ph) as usize * 4 {
                continue;
            }
            let at = |x: u32, y: u32| {
                let o = 8 + ((y * pw + x) as usize) * 4;
                f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
            };
            let map = &mut depths[k * per..(k + 1) * per];
            for y in 0..h {
                let sy = ((y as u64 * ph as u64) / h as u64) as u32;
                for x in 0..w {
                    let sx = ((x as u64 * pw as u64) / w as u64) as u32;
                    map[(y * w + x) as usize] = at(sx, sy);
                }
            }
            found += 1;
        }
        self.gs.set_depth_priors(w, h, &depths)?;
        self.depth_weight = weight;
        Ok(found)
    }

    /// Render a view with the current SH degree (row-major RGB).
    pub fn render(&mut self, view: &CameraView) -> Result<Vec<[f32; 3]>, CudaTrainError> {
        let frame = self.gs.render(view, self.cfg.background, self.active_sh)?;
        Ok(frame
            .rgb
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect())
    }

    /// Extract a mesh from the current scene (see [`crate::mesh`]), with the
    /// current SH degree; `planar` takes the median depth at the splat planes.
    pub fn extract_mesh(
        &mut self,
        planar: bool,
        views: &[&CameraView],
        support: &[[f32; 3]],
        opts: &visloc_gsplat_train::mesh::MeshOptions,
    ) -> Result<(visloc_gsplat_train::mesh::Mesh, String), CudaTrainError> {
        Ok(crate::mesh::extract_mesh(
            &mut self.gs,
            self.active_sh,
            planar,
            views,
            support,
            opts,
        )?)
    }

    /// Download the current gaussians as a [`Scene`].
    pub fn scene(&self) -> Result<Scene, CudaTrainError> {
        let packed = self.gs.scene()?;
        let group = |stride: usize, values: Vec<f32>| Group {
            stride,
            values,
            m1: Vec::new(),
            m2: Vec::new(),
        };
        let row = 3 * packed.sh_coeffs_per_channel;
        Ok(population_to_scene(
            &Population {
                transforms: group(10, packed.transforms),
                opacity: group(1, packed.opacity),
                sh: group(row, packed.sh),
            },
            self.sh_degree,
        ))
    }
}
