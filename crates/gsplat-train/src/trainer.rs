//! GPU trainer: render -> loss gradient -> backward -> Adam, all on the
//! device, with no per-step image transfer, plus Inria-style adaptive density
//! control every `densify.interval` steps (on the host, see [`crate::densify`]).
//!
//! Loss is `(1 - ssim_weight) * L1 + ssim_weight * (1 - SSIM)` (the Inria
//! loss); per-group Adam learning rates follow the Inria defaults (the mean's
//! rate is scaled by the scene extent and decays exponentially).

use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::cpu_render::Image;
use visloc_gsplat_core::gaussian::{Gaussian, Scene};
use visloc_gsplat_render::{GpuContext, GpuError, GpuScene, PackedScene, PrefixScanner, Renderer};

use crate::dataset::{load_view_rgb, Dataset, DatasetError, View};
use crate::densify::{BrushRefineConfig, DensifyConfig, DensifyReport, Group, Population};
use crate::loss::{SsimBinds, SsimKernels};

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
            ..Self::default()
        }
    }
}

/// Errors from the trainer.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Dataset(#[from] DatasetError),
    #[error("dataset has no training views")]
    NoViews,
    #[error("all views must share one resolution ({0}x{1} vs {2}x{3})")]
    MixedResolution(u32, u32, u32, u32),
}

struct AdamGroup {
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    m1: wgpu::Buffer,
    m2: wgpu::Buffer,
    n: u32,
    stride: u32,
    split_a: u32,
    split_b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamUniforms {
    n: u32,
    stride: u32,
    split_a: u32,
    split_b: u32,
    lr_a: f32,
    lr_b: f32,
    lr_c: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StatsUniforms {
    num_visible: u32,
    half_w: f32,
    half_h: f32,
    /// 0 Inria (sum of NDC xy grad norms), 1 brush (max refine weight).
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseUniforms {
    n: u32,
    seed: u32,
    scale: f32,
    max_noise: f32,
}

/// Everything sized by the gaussian count; rebuilt after densification.
struct SceneState {
    renderer: Renderer,
    loss_bind: wgpu::BindGroup,
    ssim_bind: SsimBinds,
    adam: [AdamGroup; 3],
    grad_accum: wgpu::Buffer,
    grad_count: wgpu::Buffer,
    stats_bind: wgpu::BindGroup,
    noise_bind: wgpu::BindGroup,
}

/// The trainer. Owns the renderer (and so the scene on the GPU).
pub struct Trainer {
    state: Option<SceneState>,
    cfg: TrainConfig,
    views: Vec<View>,
    width: u32,
    height: u32,
    extent: f32,
    /// brush's bound size: the median axis length of the central 75% box of
    /// the splat centres (updated at every brush refine).
    bound_size: f32,
    noise_pipeline: wgpu::ComputePipeline,
    noise_uniforms: wgpu::Buffer,
    sh_degree: u32,
    /// Ground truth of the current view (the host keeps every view packed
    /// as RGBA8 and uploads one per step: ~3 MB instead of ~300 MB resident).
    gt: wgpu::Buffer,
    gt_host: Vec<Vec<u32>>,
    loss_pipeline: wgpu::ComputePipeline,
    loss_uniforms: wgpu::Buffer,
    loss_acc: wgpu::Buffer,
    loss_steps: usize,
    ssim: SsimKernels,
    adam_pipeline: wgpu::ComputePipeline,
    stats_pipeline: wgpu::ComputePipeline,
    stats_uniforms: wgpu::Buffer,
    classify_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    /// Prefix scanner for the densify row counts, with its capacity.
    scanner: Option<(PrefixScanner, usize)>,
    step: usize,
    order: Vec<usize>,
    rng: u64,
    last_densify: Option<DensifyReport>,
    /// Per-stage wall time (ms) and count, when `GSPLAT_TRAIN_PROFILE` is set.
    profile: Option<StageTimes>,
}

/// Accumulated per-stage wall time of [`Trainer::step`] (profiling only: each
/// mark waits for the GPU, so stages are serialised).
#[derive(Debug, Default, Clone)]
pub struct StageTimes {
    last: Option<std::time::Instant>,
    pub stages: Vec<(&'static str, f64, usize)>,
}

impl StageTimes {
    fn start(&mut self) {
        self.last = Some(std::time::Instant::now());
    }
    fn mark(&mut self, dev: &wgpu::Device, name: &'static str) {
        dev.poll(wgpu::PollType::wait_indefinitely()).ok();
        let now = std::time::Instant::now();
        let ms = self
            .last
            .map(|t| (now - t).as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        self.last = Some(now);
        match self.stages.iter_mut().find(|s| s.0 == name) {
            Some(s) => {
                s.1 += ms;
                s.2 += 1;
            }
            None => self.stages.push((name, ms, 1)),
        }
    }
}

fn storage(dev: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn uniform(dev: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A compute pipeline with an auto-derived layout for `entry` in `source`.
fn pipeline(dev: &wgpu::Device, label: &str, source: &str, entry: &str) -> wgpu::ComputePipeline {
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind(
    dev: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// 2D workgroup count for `threads` 256-wide invocations (see the kernels).
fn groups_2d(threads: u32) -> (u32, u32) {
    let g = threads.div_ceil(256).max(1);
    let x = g.min(65535);
    (x, g.div_ceil(x))
}

fn pack_rgba8(rgb: &[[f32; 3]]) -> Vec<u32> {
    rgb.iter()
        .map(|p| {
            let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            c(p[0]) | (c(p[1]) << 8) | (c(p[2]) << 16) | (255 << 24)
        })
        .collect()
}

/// Blocking read of `len` f32 values from a device buffer.
fn read_f32(dev: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let bytes = (len * 4) as u64;
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_f32"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    dev.poll(wgpu::PollType::wait_indefinitely()).ok();
    let _ = rx.recv();
    let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("map")).to_vec();
    staging.unmap();
    out
}

/// Blocking read of the `index`-th u32 of a device buffer.
fn read_u32_at(dev: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, index: usize) -> u32 {
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_u32_at"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, (index * 4) as u64, &staging, 0, 4);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    dev.poll(wgpu::PollType::wait_indefinitely()).ok();
    let _ = rx.recv();
    let v = {
        let data = slice.get_mapped_range().expect("map");
        u32::from_le_bytes([data[0], data[1], data[2], data[3]])
    };
    staging.unmap();
    v
}

/// Gaussians from a population (the device parameter layouts).
fn population_to_scene(pop: &Population, degree: u32) -> Scene {
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

impl Trainer {
    /// Upload `init` and the dataset's training images and build the kernels.
    pub fn new(
        ctx: GpuContext,
        dataset: &Dataset,
        init: &Scene,
        cfg: TrainConfig,
    ) -> Result<Self, TrainError> {
        let views = dataset.train.clone();
        let first = views.first().ok_or(TrainError::NoViews)?;
        let (width, height) = (first.camera.camera.width, first.camera.camera.height);
        for v in dataset.train.iter().chain(&dataset.eval) {
            let (w, h) = (v.camera.camera.width, v.camera.camera.height);
            if (w, h) != (width, height) {
                return Err(TrainError::MixedResolution(width, height, w, h));
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

        let dev = ctx.device.clone();
        let queue = ctx.queue.clone();
        let npix = (width * height) as u64;
        let gt = storage(&dev, "gt", npix * 4);
        let mut gt_host = Vec::with_capacity(views.len());
        for v in &views {
            gt_host.push(pack_rgba8(&load_view_rgb(v)?));
        }
        let loss_pipeline = pipeline(
            &dev,
            "loss_l1",
            include_str!("shaders/loss_l1.wgsl"),
            "loss_l1",
        );
        let loss_uniforms = uniform(&dev, "loss_uniforms", 16);
        let l1_weight = 1.0 - cfg.ssim_weight;
        queue.write_buffer(
            &loss_uniforms,
            0,
            bytemuck::cast_slice(&[npix as u32, l1_weight.to_bits(), 0, 0]),
        );
        let loss_acc = storage(&dev, "loss_acc", 4);
        let ssim = SsimKernels::new(&dev, width, height);
        ssim.set_weight(&queue, cfg.ssim_weight);
        let adam_pipeline = pipeline(&dev, "adam", include_str!("shaders/adam.wgsl"), "adam");
        let stats_pipeline = pipeline(
            &dev,
            "densify_stats",
            include_str!("shaders/densify_stats.wgsl"),
            "densify_stats",
        );
        let stats_uniforms = uniform(&dev, "stats_uniforms", 16);
        let noise_pipeline = pipeline(
            &dev,
            "mean_noise",
            include_str!("shaders/mean_noise.wgsl"),
            "mean_noise",
        );
        let noise_uniforms = uniform(&dev, "noise_uniforms", 16);
        let init_means: Vec<[f32; 3]> = init
            .gaussians
            .iter()
            .map(|g| [g.mean.x, g.mean.y, g.mean.z])
            .collect();
        let bound_size = central_bound_size(&init_means, 0.75).unwrap_or(extent);
        let classify_pipeline = pipeline(
            &dev,
            "densify_classify",
            include_str!("shaders/densify_classify.wgsl"),
            "densify_classify",
        );
        let scatter_pipeline = pipeline(
            &dev,
            "densify_scatter",
            include_str!("shaders/densify_scatter.wgsl"),
            "densify_scatter",
        );

        let mut trainer = Self {
            state: None,
            views,
            width,
            height,
            extent,
            bound_size,
            noise_pipeline,
            noise_uniforms,
            sh_degree: init.sh_degree,
            gt,
            gt_host,
            loss_pipeline,
            loss_uniforms,
            loss_acc,
            loss_steps: 0,
            ssim,
            adam_pipeline,
            stats_pipeline,
            stats_uniforms,
            classify_pipeline,
            scatter_pipeline,
            scanner: None,
            step: 0,
            order: Vec::new(),
            rng: cfg.seed.max(1),
            cfg,
            last_densify: None,
            profile: std::env::var("GSPLAT_TRAIN_PROFILE")
                .is_ok()
                .then(StageTimes::default),
        };
        trainer.order = (0..trainer.views.len()).collect();
        let renderer = Renderer::from_packed(ctx, PackedScene::from_scene(init), width, height)?;
        trainer.build_state(renderer, None)?;
        Ok(trainer)
    }

    /// (Re)build everything sized by the gaussian count around `renderer`.
    /// `moments` carries the Adam state (m1, m2 per group, already on the
    /// device) across densification; `None` starts from zero.
    fn build_state(
        &mut self,
        mut renderer: Renderer,
        moments: Option<[(wgpu::Buffer, wgpu::Buffer); 3]>,
    ) -> Result<(), TrainError> {
        let n = renderer.num_gaussians() as u32;
        let cpc2 = (self.sh_degree + 1) * (self.sh_degree + 1);
        renderer.set_skip_readback(true);
        // Adam zeroes each gradient after reading it.
        renderer.set_grads_zeroed_by_caller(true);
        let dev = renderer.ctx.device.clone();

        let out_img = renderer.output_buffer().clone();
        let d_image = renderer.d_image_buffer()?.clone();
        let loss_bind = bind(
            &dev,
            &self.loss_pipeline,
            "loss",
            &[
                &self.loss_uniforms,
                &out_img,
                &self.gt,
                &d_image,
                &self.loss_acc,
            ],
        );
        let ssim_bind = self.ssim.bind(&dev, &out_img, &self.gt, &d_image);

        let params = renderer.param_buffers();
        let (pt, po, ps) = (
            params.transforms.clone(),
            params.opacity.clone(),
            params.sh.clone(),
        );
        let grads = renderer.grad_buffers()?;
        let (gt, go, gs) = (
            grads.transforms.clone(),
            grads.opacity.clone(),
            grads.sh.clone(),
        );
        let mk_group = |label: &str,
                        p: &wgpu::Buffer,
                        g: &wgpu::Buffer,
                        stride: u32,
                        a: u32,
                        b: u32,
                        init: Option<(wgpu::Buffer, wgpu::Buffer)>| {
            let len = n * stride;
            let (m1, m2) = init.unwrap_or_else(|| {
                (
                    storage(&dev, label, len as u64 * 4),
                    storage(&dev, label, len as u64 * 4),
                )
            });
            let u = uniform(&dev, label, std::mem::size_of::<AdamUniforms>() as u64);
            let bg = bind(&dev, &self.adam_pipeline, label, &[&u, p, g, &m1, &m2]);
            AdamGroup {
                uniforms: u,
                bind: bg,
                m1,
                m2,
                n: len,
                stride,
                split_a: a,
                split_b: b,
            }
        };
        let [mt, mo, ms] = match moments {
            Some([a, b, c]) => [Some(a), Some(b), Some(c)],
            None => [None, None, None],
        };
        let adam = [
            // mean | quat | log-scale
            mk_group("adam_t", &pt, &gt, 10, 3, 7, mt),
            mk_group("adam_o", &po, &go, 1, 1, 1, mo),
            // DC | rest
            mk_group("adam_sh", &ps, &gs, 3 * cpc2, 3, 3 * cpc2, ms),
        ];

        let grad_accum = storage(&dev, "grad_accum", n as u64 * 4);
        let grad_count = storage(&dev, "grad_count", n as u64 * 4);
        let (gfc, screen, _) = renderer.screen_grad_buffers()?;
        let stats_bind = bind(
            &dev,
            &self.stats_pipeline,
            "stats",
            &[&self.stats_uniforms, gfc, screen, &grad_accum, &grad_count],
        );
        let noise_bind = bind(
            &dev,
            &self.noise_pipeline,
            "mean_noise",
            &[&self.noise_uniforms, &pt, &po],
        );
        self.state = Some(SceneState {
            renderer,
            loss_bind,
            ssim_bind,
            adam,
            grad_accum,
            grad_count,
            stats_bind,
            noise_bind,
        });
        Ok(())
    }

    fn st(&self) -> &SceneState {
        self.state.as_ref().expect("trainer state")
    }

    /// Per-stage times accumulated so far (`GSPLAT_TRAIN_PROFILE` only), and
    /// reset them.
    pub fn take_profile(&mut self) -> Option<StageTimes> {
        self.profile.as_mut().map(std::mem::take)
    }

    /// Steps taken so far.
    pub fn steps_done(&self) -> usize {
        self.step
    }

    /// Current number of gaussians.
    pub fn num_gaussians(&self) -> usize {
        self.st().renderer.num_gaussians()
    }

    /// Scene extent used to scale the mean learning rate.
    pub fn extent(&self) -> f32 {
        self.extent
    }

    /// The report of the densification run by the last [`Trainer::step`], if any.
    pub fn take_densify_report(&mut self) -> Option<DensifyReport> {
        self.last_densify.take()
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
    pub fn step(&mut self) -> Result<(), TrainError> {
        let vi = self.next_view();
        let view = self.views[vi].camera;
        let bg = self.cfg.background;
        let brush = self.cfg.brush_refine.is_some();
        // brush gathers refine statistics for the whole run (pruning never
        // stops); the Inria rule only until densification stops.
        let densifying = brush
            || self
                .cfg
                .densify
                .as_ref()
                .is_some_and(|d| self.step < d.stop);
        let npix = self.width * self.height;
        let (half_w, half_h) = (self.width as f32 * 0.5, self.height as f32 * 0.5);

        let st = self.state.as_mut().expect("trainer state");
        let dev = st.renderer.ctx.device.clone();
        if let Some(p) = self.profile.as_mut() {
            dev.poll(wgpu::PollType::wait_indefinitely()).ok();
            p.start();
        }
        let active_sh = (self.cfg.sh_degree_interval > 0)
            .then(|| (self.step / self.cfg.sh_degree_interval) as u32);
        st.renderer.set_active_sh_degree(active_sh);
        let _ = st.renderer.render(&view, bg);
        if let Some(p) = self.profile.as_mut() {
            p.mark(&dev, "forward");
        }
        let queue = st.renderer.ctx.queue.clone();
        // Ordered after the previous step's loss work, before this one's.
        queue.write_buffer(&self.gt, 0, bytemuck::cast_slice(&self.gt_host[vi]));
        {
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loss"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.loss_pipeline);
                pass.set_bind_group(0, &st.loss_bind, &[]);
                let (x, y) = groups_2d(npix);
                pass.dispatch_workgroups(x, y, 1);
                if self.cfg.ssim_weight > 0.0 {
                    self.ssim.encode(&mut pass, &st.ssim_bind);
                }
            }
            queue.submit(Some(enc.finish()));
        }
        self.loss_steps += 1;
        if let Some(p) = self.profile.as_mut() {
            p.mark(&dev, "loss");
        }

        st.renderer.backward_on_device()?;
        if let Some(p) = self.profile.as_mut() {
            p.mark(&dev, "backward");
        }

        // Densification statistics, then Adam (bias-corrected, mean-LR decay).
        let t = (self.step + 1) as f32;
        let (beta1, beta2) = (0.9f32, 0.999f32);
        let frac = (self.step as f32 / self.cfg.steps.max(1) as f32).min(1.0);
        let lr_mean = (self.cfg.lr_mean.ln() * (1.0 - frac) + self.cfg.lr_mean_final.ln() * frac)
            .exp()
            * if brush { self.bound_size } else { self.extent };
        let lr_scale = match self.cfg.lr_scale_final {
            Some(end) => (self.cfg.lr_scale.ln() * (1.0 - frac) + end.ln() * frac).exp(),
            None => self.cfg.lr_scale,
        };
        let lrs = [
            (lr_mean, self.cfg.lr_quat, lr_scale),
            (
                self.cfg.lr_opacity,
                self.cfg.lr_opacity,
                self.cfg.lr_opacity,
            ),
            (self.cfg.lr_sh_dc, self.cfg.lr_sh_rest, self.cfg.lr_sh_rest),
        ];
        let nv = st.renderer.screen_grad_buffers()?.2 as u32;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adam"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            if densifying && nv > 0 {
                let su = StatsUniforms {
                    num_visible: nv,
                    half_w,
                    half_h,
                    mode: brush as u32,
                };
                queue.write_buffer(&self.stats_uniforms, 0, bytemuck::bytes_of(&su));
                pass.set_pipeline(&self.stats_pipeline);
                pass.set_bind_group(0, &st.stats_bind, &[]);
                pass.dispatch_workgroups(nv.div_ceil(256), 1, 1);
            }
            pass.set_pipeline(&self.adam_pipeline);
            for (g, (a, b, c)) in st.adam.iter().zip(lrs) {
                let u = AdamUniforms {
                    n: g.n,
                    stride: g.stride,
                    split_a: g.split_a,
                    split_b: g.split_b,
                    lr_a: a,
                    lr_b: b,
                    lr_c: c,
                    beta1,
                    beta2,
                    eps: 1e-15,
                    bc1: 1.0 - beta1.powf(t),
                    bc2: 1.0 - beta2.powf(t),
                };
                queue.write_buffer(&g.uniforms, 0, bytemuck::bytes_of(&u));
                pass.set_bind_group(0, &g.bind, &[]);
                let (x, y) = groups_2d(g.n);
                pass.dispatch_workgroups(x, y, 1);
            }
            if let Some(b) = self.cfg.brush_refine.as_ref() {
                // Mean noise on nearly transparent gaussians, after the step.
                let n = st.renderer.num_gaussians() as u32;
                let nu = NoiseUniforms {
                    n,
                    seed: (self.cfg.seed as u32) ^ (self.step as u32).wrapping_mul(0x85EB_CA6B),
                    scale: lr_mean * b.mean_noise_weight,
                    max_noise: 0.25 * self.bound_size,
                };
                queue.write_buffer(&self.noise_uniforms, 0, bytemuck::bytes_of(&nu));
                pass.set_pipeline(&self.noise_pipeline);
                pass.set_bind_group(0, &st.noise_bind, &[]);
                let (x, y) = groups_2d(n);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        queue.submit(Some(enc.finish()));
        if let Some(p) = self.profile.as_mut() {
            p.mark(&dev, "adam");
        }
        self.step += 1;

        if let Some(b) = self.cfg.brush_refine.clone() {
            if self.step % b.refine_every.max(1) == 0 {
                self.brush_refine_now(&b)?;
                let dev = self.st().renderer.ctx.device.clone();
                if let Some(p) = self.profile.as_mut() {
                    p.mark(&dev, "densify");
                }
            }
        } else if let Some(d) = self.cfg.densify.clone() {
            let s = self.step;
            let at_densify = s >= d.start && s <= d.stop && s % d.interval == 0;
            let at_reset = s < d.stop && s % d.opacity_reset_interval == 0;
            if at_densify || at_reset {
                self.densify_now(&d, at_densify, at_reset)?;
                let dev = self.st().renderer.ctx.device.clone();
                if let Some(p) = self.profile.as_mut() {
                    p.mark(&dev, "densify");
                }
            }
        }
        Ok(())
    }

    /// Densify (and/or reset opacity) on the device: classify every gaussian,
    /// prefix-sum the output row counts, scatter parameters and Adam moments
    /// into new buffers, and rebuild the renderer around them. No host copy of
    /// the scene is made.
    fn densify_now(
        &mut self,
        d: &DensifyConfig,
        grow: bool,
        reset: bool,
    ) -> Result<(), TrainError> {
        let t0 = std::time::Instant::now();
        let st = self.state.as_ref().expect("trainer state");
        let dev = st.renderer.ctx.device.clone();
        let queue = st.renderer.ctx.queue.clone();
        let n = st.renderer.num_gaussians();
        // Population cap: past it only pruning (and opacity resets) run.
        let grow = grow && n < d.max_gaussians;
        let params = st.renderer.param_buffers();
        let src = [
            params.transforms.clone(),
            params.opacity.clone(),
            params.sh.clone(),
        ];

        // 1. Classify.
        let actions = storage(&dev, "densify_actions", n as u64 * 4);
        let counts = storage(&dev, "densify_counts", n as u64 * 4);
        let tallies = storage(&dev, "densify_tallies", 12);
        let cu = uniform(&dev, "densify_classify_u", 32);
        let words: [u32; 8] = [
            n as u32,
            grow as u32,
            (self.step > d.opacity_reset_interval) as u32,
            0,
            d.grad_threshold.to_bits(),
            (d.percent_dense * self.extent).to_bits(),
            d.min_opacity.to_bits(),
            (d.max_world_scale * self.extent).to_bits(),
        ];
        queue.write_buffer(&cu, 0, bytemuck::cast_slice(&words));
        let cb = bind(
            &dev,
            &self.classify_pipeline,
            "densify_classify",
            &[
                &cu,
                &src[0],
                &src[1],
                &st.grad_accum,
                &st.grad_count,
                &actions,
                &counts,
                &tallies,
            ],
        );
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.classify_pipeline);
            pass.set_bind_group(0, &cb, &[]);
            let (x, y) = groups_2d(n as u32);
            pass.dispatch_workgroups(x, y, 1);
        }
        queue.submit(Some(enc.finish()));

        let n_after = self.apply_actions(n, &actions, &counts, reset)?;
        let st = self.state.as_ref().expect("trainer state");
        let (dev, queue) = (
            st.renderer.ctx.device.clone(),
            st.renderer.ctx.queue.clone(),
        );
        let t: Vec<u32> = read_f32(&dev, &queue, &tallies, 3)
            .iter()
            .map(|x| x.to_bits())
            .collect();
        if grow {
            self.last_densify = Some(DensifyReport {
                cloned: t[0] as usize,
                split: t[1] as usize,
                pruned: t[2] as usize,
                before: n,
                after: n_after,
            });
        }

        if self.profile.is_some() && self.step % 1000 == 0 {
            eprintln!(
                "[densify] {n} -> {n_after} on device in {:.0} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(())
    }

    /// brush's refine step (see [`BrushRefineConfig`]): decide on the host
    /// which gaussians to prune and which to split (weighted sampling, as
    /// brush does), then apply the actions on the device.
    fn brush_refine_now(&mut self, b: &BrushRefineConfig) -> Result<(), TrainError> {
        let st = self.state.as_ref().expect("trainer state");
        let dev = st.renderer.ctx.device.clone();
        let queue = st.renderer.ctx.queue.clone();
        let n = st.renderer.num_gaussians();
        if n == 0 {
            return Ok(());
        }
        let params = st.renderer.param_buffers();
        let transforms = read_f32(&dev, &queue, params.transforms, n * 10);
        let opacity_logit = read_f32(&dev, &queue, params.opacity, n);
        let weight_max = read_f32(&dev, &queue, &st.grad_accum, n);
        let seen = read_f32(&dev, &queue, &st.grad_count, n);

        // Bounds of the current splat centres (brush: central 75% box).
        let means: Vec<[f32; 3]> = transforms
            .chunks_exact(10)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        let (center, size) = central_bounds(&means, 0.75).unwrap_or(([0.0; 3], self.bound_size));
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let opacity: Vec<f32> = opacity_logit.iter().map(|&x| sigmoid(x)).collect();
        let pruned: Vec<bool> = (0..n)
            .map(|i| {
                let t = &transforms[i * 10..i * 10 + 10];
                opacity[i] < b.min_opacity
                    || t[7..10].iter().any(|&ls| ls < -15.0)
                    || (0..3).any(|k| (t[k] - center[k]).abs() > size * 10.0)
            })
            .collect();
        let pruned_count = pruned.iter().filter(|p| **p).count();

        // Replacements for the pruned, sampled by opacity; then growth from
        // the gaussians above the refine threshold, sampled by their weight.
        let mut add = vec![false; n];
        let rng = &mut self.rng;
        let mut sample = |weights: &[f32], count: usize, add: &mut Vec<bool>| {
            let mut cdf = Vec::with_capacity(weights.len());
            let mut acc = 0.0f64;
            for &w in weights {
                acc += w.max(0.0) as f64;
                cdf.push(acc);
            }
            if acc <= 0.0 {
                return;
            }
            for _ in 0..count {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                let u = (*rng >> 11) as f64 / (1u64 << 53) as f64 * acc;
                let j = cdf.partition_point(|&c| c <= u).min(weights.len() - 1);
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
                .map(|i| !pruned[i] && weight_max[i] / seen[i].max(1.0) > b.growth_grad_threshold)
                .collect();
            let threshold_count = above.iter().filter(|a| **a).count();
            let grow = ((threshold_count as f32 * b.growth_select_fraction).round() as usize)
                .saturating_sub(pruned_count);
            let current = n - pruned_count + add.iter().filter(|a| **a).count();
            let grow = grow.min(b.max_gaussians.saturating_sub(current));
            if grow > 0 {
                let w: Vec<f32> = (0..n)
                    .map(|i| if above[i] { weight_max[i] } else { 0.0 })
                    .collect();
                let before = add.iter().filter(|a| **a).count();
                sample(&w, grow, &mut add);
                grown = add.iter().filter(|a| **a).count() - before;
            }
        }
        let actions_host: Vec<u32> = (0..n)
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
        let counts_host: Vec<u32> = actions_host
            .iter()
            .map(|&a| match a {
                0 => 0,
                4 => 2,
                _ => 1,
            })
            .collect();
        let actions = storage(&dev, "refine_actions", n as u64 * 4);
        let counts = storage(&dev, "refine_counts", n as u64 * 4);
        queue.write_buffer(&actions, 0, bytemuck::cast_slice(&actions_host));
        queue.write_buffer(&counts, 0, bytemuck::cast_slice(&counts_host));
        let n_after = self.apply_actions(n, &actions, &counts, false)?;
        self.bound_size = size;
        self.last_densify = Some(DensifyReport {
            cloned: grown,
            split: actions_host.iter().filter(|a| **a == 4).count() - grown,
            pruned: pruned_count,
            before: n,
            after: n_after,
        });
        Ok(())
    }

    /// Apply per-gaussian actions (0 prune, 1 keep, 2 clone, 3 Inria split,
    /// 4 brush split) with their output row counts: prefix-sum the counts,
    /// scatter parameters and Adam moments into new buffers, and rebuild the
    /// renderer around them. Returns the new gaussian count.
    fn apply_actions(
        &mut self,
        n: usize,
        actions: &wgpu::Buffer,
        counts: &wgpu::Buffer,
        reset: bool,
    ) -> Result<usize, TrainError> {
        let st = self.state.as_ref().expect("trainer state");
        let dev = st.renderer.ctx.device.clone();
        let queue = st.renderer.ctx.queue.clone();
        let params = st.renderer.param_buffers();
        let src = [
            params.transforms.clone(),
            params.opacity.clone(),
            params.sh.clone(),
        ];
        let cum = storage(&dev, "densify_cum", n as u64 * 4);
        // 2. Prefix sum of the row counts -> output offsets and the new count.
        if self.scanner.as_ref().is_none_or(|(_, cap)| *cap < n) {
            self.scanner = Some((PrefixScanner::new(&dev, n.max(1)), n));
        }
        let (scanner, _) = self.scanner.as_ref().expect("scanner");
        scanner.scan(&dev, &queue, counts, &cum, n);
        let n_after = if n > 0 {
            read_u32_at(&dev, &queue, &cum, n - 1) as usize
        } else {
            0
        };
        // 3. Scatter every group into new buffers.
        let st = self.state.as_ref().expect("trainer state");
        let cpc2 = (self.sh_degree + 1) * (self.sh_degree + 1);
        let strides = [10u32, 1, 3 * cpc2];
        let reset_cap = (0.01f32 / 0.99).ln();
        let seed = (self.cfg.seed as u32) ^ (self.step as u32).wrapping_mul(0x9E37_79B9);
        let mut out: Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> = Vec::with_capacity(3);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut keep_alive = Vec::new();
        for (kind, stride) in strides.iter().enumerate() {
            let len = (n_after as u64) * (*stride as u64) * 4;
            let dp = storage(&dev, "params", len);
            let dm1 = storage(&dev, "adam_m1", len);
            let dm2 = storage(&dev, "adam_m2", len);
            let su = uniform(&dev, "densify_scatter_u", 32);
            let words: [u32; 8] = [
                n as u32,
                *stride,
                kind as u32,
                (reset && kind == 1) as u32,
                reset_cap.to_bits(),
                seed,
                0,
                0,
            ];
            queue.write_buffer(&su, 0, bytemuck::cast_slice(&words));
            let g = &st.adam[kind];
            let sb = bind(
                &dev,
                &self.scatter_pipeline,
                "densify_scatter",
                &[
                    &su, actions, &cum, &src[kind], &g.m1, &g.m2, &dp, &dm1, &dm2,
                ],
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &sb, &[]);
                let (x, y) = groups_2d(n as u32);
                pass.dispatch_workgroups(x, y, 1);
            }
            keep_alive.push((su, sb));
            out.push((dp, dm1, dm2));
        }
        queue.submit(Some(enc.finish()));
        drop(keep_alive);

        // 4. Rebuild around the new buffers (no upload).
        let ctx = self
            .state
            .take()
            .expect("trainer state")
            .renderer
            .into_context();
        let mut it = out.into_iter();
        let (pt, mt1, mt2) = it.next().expect("transforms");
        let (po, mo1, mo2) = it.next().expect("opacity");
        let (ps, ms1, ms2) = it.next().expect("sh");
        let scene = GpuScene::from_buffers(pt, po, ps, n_after, self.sh_degree);
        let renderer = Renderer::from_gpu_scene(ctx, scene, self.width, self.height)?;
        self.build_state(renderer, Some([(mt1, mt2), (mo1, mo2), (ms1, ms2)]))?;
        Ok(n_after)
    }

    /// Mean L1 loss over the steps since the last call (reads the device).
    pub fn take_mean_loss(&mut self) -> f32 {
        let st = self.st();
        let (dev, queue) = (&st.renderer.ctx.device, &st.renderer.ctx.queue);
        let v = read_f32(dev, queue, &self.loss_acc, 1);
        let bits = v.first().map(|x| x.to_bits()).unwrap_or(0);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&self.loss_acc, 0, None);
        queue.submit(Some(enc.finish()));
        let steps = self.loss_steps.max(1);
        self.loss_steps = 0;
        bits as f32 * 1e-6 / steps as f32
    }

    /// Render a view (with image readback), e.g. for evaluation.
    pub fn render(&mut self, view: &CameraView) -> Image {
        let bg = self.cfg.background;
        let r = &mut self.state.as_mut().expect("trainer state").renderer;
        r.set_skip_readback(false);
        let img = r.render(view, bg);
        r.set_skip_readback(true);
        img
    }

    /// Download the current gaussians as a [`Scene`].
    /// GPU memory summary from the allocator (backends that expose one):
    /// allocated / reserved MiB and the largest buffers grouped by label.
    pub fn memory_report(&self) -> Option<String> {
        let dev = &self.st().renderer.ctx.device;
        // Let wgpu release buffers dropped since the last maintain.
        dev.poll(wgpu::PollType::wait_indefinitely()).ok();
        let report = dev.generate_allocator_report()?;
        let mut by_name: std::collections::BTreeMap<String, (u64, usize)> = Default::default();
        for a in &report.allocations {
            let e = by_name.entry(a.name.clone()).or_default();
            e.0 += a.size;
            e.1 += 1;
        }
        let mut top: Vec<(String, (u64, usize))> = by_name.into_iter().collect();
        top.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        let mut line = format!(
            "gpu mem: allocated {:.0} MiB, reserved {:.0} MiB |",
            mib(report.total_allocated_bytes),
            mib(report.total_reserved_bytes)
        );
        for (name, (bytes, count)) in top.iter().take(12) {
            line.push_str(&format!(" {name} {:.0}x{count}", mib(*bytes)));
        }
        Some(line)
    }

    pub fn scene(&self) -> Scene {
        let (t, o, sh) = self.st().renderer.read_params();
        let group = |stride: usize, values: Vec<f32>| Group {
            stride,
            m1: Vec::new(),
            m2: Vec::new(),
            values,
        };
        let cpc2 = ((self.sh_degree + 1) * (self.sh_degree + 1)) as usize;
        population_to_scene(
            &Population {
                transforms: group(10, t),
                opacity: group(1, o),
                sh: group(3 * cpc2, sh),
            },
            self.sh_degree,
        )
    }
}

/// brush's bounds: the per-axis central `percentile` box of `points`;
/// returns (centre, median axis length).
fn central_bounds(points: &[[f32; 3]], percentile: f32) -> Option<([f32; 3], f32)> {
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

fn central_bound_size(points: &[[f32; 3]], percentile: f32) -> Option<f32> {
    central_bounds(points, percentile).map(|(_, s)| s)
}
