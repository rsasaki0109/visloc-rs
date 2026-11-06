//! VLFeat-compatible SIFT on the GPU: the extractor `gsplat_photos` runs
//! (`SiftConfig::vlfeat_compatible_detector` and
//! `vlfeat_compatible_descriptor`), in f32 WGSL.
//!
//! The scale spaces, the DoG extrema with VLFeat's five-step refinement, the
//! orientation histograms, the descriptor pyramid's gradients and the
//! descriptors run on the GPU. The host keeps what decides which keypoints
//! survive and in what order, exactly as the CPU path does: the candidates'
//! scan order, the per-locus orientation cap, the per-level feature cap
//! (`max_keypoints`) and each keypoint's descriptor level (in f64).
//!
//! It is a port, not a bit-exact one: the CPU path computes in f64, so a
//! handful of borderline extrema and orientation peaks differ and the
//! descriptor bytes can differ by one. `examples/sift_vlfeat_gpu_bench.rs`
//! compares the two on a directory of images.
//!
//! Every pyramid level is its own buffer, so no binding is larger than one
//! level (devices may cap a binding at 128 MiB). Buffers, dispatches and
//! bind groups are built once per image size and reused.

use std::collections::HashMap;
use std::num::NonZeroU64;

use visloc_gsplat_render::GpuContext;
use visloc_vision::features::sift::{
    GrayImage, SiftConfig, SiftDetector, SiftKeypoint, SiftNormalization,
};

use crate::extractor::{mapped, storage, u32_words, SiftGpuError};

/// Uniform parameter slot stride and binding size.
const SLOT: u64 = 256;
const SLOT_BIND: u64 = 64;
/// Words per candidate (`vlfeat_detect.wgsl`).
const CAND_WORDS: usize = 16;
/// Words per keypoint handed to `vl_describe`.
const KP_WORDS: usize = 16;
/// Initial candidate capacity per octave (grown when an image overflows it).
const INITIAL_CAP: usize = 1 << 16;

struct Pipes {
    upsample: wgpu::ComputePipeline,
    blur_h: wgpu::ComputePipeline,
    blur_v: wgpu::ComputePipeline,
    halve: wgpu::ComputePipeline,
    dog: wgpu::ComputePipeline,
    extrema: wgpu::ComputePipeline,
    orient_args: wgpu::ComputePipeline,
    orient: wgpu::ComputePipeline,
    gradient: wgpu::ComputePipeline,
    describe: wgpu::ComputePipeline,
}

impl Pipes {
    /// In `Op::pipe` order.
    fn list(&self) -> [&wgpu::ComputePipeline; 9] {
        [
            &self.upsample,
            &self.blur_h,
            &self.blur_v,
            &self.halve,
            &self.dog,
            &self.extrema,
            &self.orient,
            &self.gradient,
            &self.orient_args,
        ]
    }
}

/// The settings a [`Plan`] depends on (besides the image size).
#[derive(Clone, Copy, PartialEq)]
struct PlanKey {
    width: usize,
    height: usize,
    cap: usize,
    octaves: usize,
    intervals: usize,
    sigma_base: f64,
    sigma_input: f64,
    contrast_threshold: f64,
    edge_threshold: f64,
    bilinear: bool,
}

impl PlanKey {
    fn new(width: usize, height: usize, cap: usize, config: &SiftConfig) -> Self {
        Self {
            width,
            height,
            cap,
            octaves: config.octaves,
            intervals: config.intervals.max(1),
            sigma_base: config.sigma_base,
            sigma_input: config.sigma_input,
            contrast_threshold: config.contrast_threshold,
            edge_threshold: config.edge_threshold,
            bilinear: config.vlfeat_bilinear_orientations,
        }
    }
}

/// Everything one image size needs, built once: the buffers, the recorded
/// dispatches of the detection and descriptor scale spaces, and their bind
/// groups.
struct Plan {
    key: PlanKey,
    det_dims: Vec<(usize, usize)>,
    desc_dims: Vec<(usize, usize)>,
    img: wgpu::Buffer,
    /// One candidate list per detection octave.
    cand: Vec<wgpu::Buffer>,
    counters: wgpu::Buffer,
    /// `grad[o][l - 1]`: gradients of descriptor level `l` (1..=intervals)
    /// of octave `o`, the levels a keypoint can select.
    grad: Vec<Vec<wgpu::Buffer>>,
    /// `vl_orient`'s indirect dispatch sizes, three words per octave.
    args: wgpu::Buffer,
    /// Host-readable copy of the counters and candidates.
    staging: wgpu::Buffer,
    /// `vl_describe`'s buffers for this plan's gradients (grown on demand).
    desc: Option<DescBuffers>,
    // Kept alive for the bind groups.
    _levels: Vec<wgpu::Buffer>,
    _params: wgpu::Buffer,
    passes: Vec<(usize, wgpu::BindGroup, Dispatch)>,
}

/// How an op is dispatched: a fixed grid, or the grid an earlier op wrote
/// at this byte offset of `Plan::args`.
#[derive(Clone, Copy)]
enum Dispatch {
    Grid(u32, u32, u32),
    Indirect(u64),
}

/// Grown on demand: the keypoints `vl_describe` reads, the bytes it writes
/// and one bind group per gradient level.
struct DescBuffers {
    capacity: usize,
    params: wgpu::Buffer,
    kps: wgpu::Buffer,
    out: wgpu::Buffer,
    /// Host-readable copy of `out`.
    staging: wgpu::Buffer,
    /// `binds[o * intervals + l - 1]`, with uniform slot of the same index.
    binds: Vec<wgpu::BindGroup>,
}

/// The large buffers of one plan slot: the image, the scale spaces'
/// levels, the gradients, the candidates, the descriptors and their
/// staging copies. They outlive the plans that bind them, and one is only
/// reallocated when a plan needs it larger, so a stream that alternates
/// image sizes (a rig's cameras) records new dispatches and bind groups
/// (well under a millisecond) instead of allocating the buffers again
/// (about 100 ms at 3840x2160), and holds the memory of its largest plan.
#[derive(Default)]
struct Pool {
    buffers: HashMap<(&'static str, usize, usize), wgpu::Buffer>,
    /// The buffers allocated so far, in order (for tests).
    allocated: Vec<(&'static str, usize, usize)>,
}

impl Pool {
    /// Buffer `(label, i, j)` with at least `bytes`, from `make` when the
    /// kept one is smaller (or there is none).
    fn get(
        &mut self,
        (label, i, j): (&'static str, usize, usize),
        bytes: u64,
        make: impl FnOnce(&'static str, u64) -> wgpu::Buffer,
    ) -> wgpu::Buffer {
        match self.buffers.get(&(label, i, j)) {
            Some(buffer) if buffer.size() >= bytes => buffer.clone(),
            _ => {
                let buffer = make(label, bytes);
                self.allocated.push((label, i, j));
                self.buffers.insert((label, i, j), buffer.clone());
                buffer
            }
        }
    }

    fn storage(
        &mut self,
        dev: &wgpu::Device,
        key: (&'static str, usize, usize),
        bytes: u64,
    ) -> wgpu::Buffer {
        self.get(key, bytes, |label, bytes| {
            storage(dev, label, bytes, wgpu::BufferUsages::empty())
        })
    }

    fn staging(
        &mut self,
        dev: &wgpu::Device,
        key: (&'static str, usize, usize),
        bytes: u64,
    ) -> wgpu::Buffer {
        self.get(key, bytes, |_, bytes| staging_buffer(dev, bytes))
    }
}

/// VLFeat-compatible GPU SIFT extractor (owns its [`GpuContext`]).
///
/// Its buffers are sized for the largest image it has processed and kept
/// across image sizes (see `Pool`); [`Self::clear_plan_cache`] frees them.
pub struct VlfeatSiftGpu {
    ctx: GpuContext,
    pipes: Pipes,
    cap: usize,
    /// One plan for `extract`; `extract_batch` alternates two, so one image
    /// is detected while the host handles the previous one.
    plans: Vec<Plan>,
    /// `pools[p]`: the buffers of `plans[p]`, kept across keys.
    pools: Vec<Pool>,
}

/// One refined DoG extremum read back from the GPU.
#[derive(Clone, Copy)]
struct Candidate {
    octave: usize,
    ox: u32,
    oy: u32,
    level: u32,
    x: f32,
    y: f32,
    scale: f32,
    orientations: usize,
    angles: [f32; 4],
}

/// The keypoint fields `vl_describe` reads (`DescKp` in
/// `vlfeat_describe.wgsl`), and the gradient level they are read from.
#[derive(Clone, Copy)]
struct DescKp {
    octave: usize,
    level: usize,
    lw: u32,
    xi: i32,
    yi: i32,
    min_dx: i32,
    max_dx: i32,
    min_dy: i32,
    max_dy: i32,
    off_x: f32,
    off_y: f32,
    sbp: f32,
    cos_o: f32,
    sin_o: f32,
    orientation: f32,
}

impl DescKp {
    fn words(&self) -> [u32; KP_WORDS] {
        [
            0,
            self.lw,
            self.xi as u32,
            self.yi as u32,
            self.min_dx as u32,
            self.max_dx as u32,
            self.min_dy as u32,
            self.max_dy as u32,
            self.off_x.to_bits(),
            self.off_y.to_bits(),
            self.sbp.to_bits(),
            self.cos_o.to_bits(),
            self.sin_o.to_bits(),
            self.orientation.to_bits(),
            0,
            0,
        ]
    }
}

/// One recorded dispatch: pipe 0 upsample, 1 blur_h, 2 blur_v, 3 halve,
/// 4 DoG, 5 extrema, 6 orientations, 7 gradients, 8 orientation grid;
/// `(binding, buffer)`s.
struct Op<'a> {
    pipe: usize,
    slot: u64,
    bufs: Vec<(u32, &'a wgpu::Buffer)>,
    groups: Dispatch,
}

/// Uniform slots, one per op.
#[derive(Default)]
struct Slots {
    words: Vec<u32>,
}

impl Slots {
    fn push(&mut self, fields: &[u32]) -> u64 {
        let slot = (self.words.len() * 4) as u64;
        let mut s = [0u32; (SLOT / 4) as usize];
        s[..fields.len()].copy_from_slice(fields);
        self.words.extend_from_slice(&s);
        slot
    }
}

fn groups2d(w: usize, h: usize) -> Dispatch {
    Dispatch::Grid((w as u32).div_ceil(16), (h as u32).div_ceil(16), 1)
}

/// A separable blur of `src` into `dst` through `tmp`, with the kernel at
/// `(weights offset, radius)`.
fn blur_ops<'a>(
    ops: &mut Vec<Op<'a>>,
    slots: &mut Slots,
    (w, h): (usize, usize),
    src: &'a wgpu::Buffer,
    tmp: &'a wgpu::Buffer,
    dst: &'a wgpu::Buffer,
    (wt_off, radius): (u32, u32),
) {
    let (wu, hu) = (w as u32, h as u32);
    let slot = slots.push(&[wu, hu, 0, 0, radius, wt_off, wu, hu]);
    ops.push(Op {
        pipe: 1,
        slot,
        bufs: vec![(1, src), (2, tmp)],
        groups: groups2d(w, h),
    });
    let slot = slots.push(&[wu, hu, 0, 0, radius, wt_off, wu, hu]);
    ops.push(Op {
        pipe: 2,
        slot,
        bufs: vec![(1, tmp), (2, dst)],
        groups: groups2d(w, h),
    });
}

/// `gaussian_kernel_vlfeat`: four-sigma taps, normalized in f64.
fn gaussian_kernel_vlfeat(sigma: f64) -> Vec<f32> {
    let sigma = sigma.max(1e-6);
    let radius = (sigma * 4.0).ceil().max(1.0) as i64;
    let denom = 2.0 * sigma * sigma;
    let raw: Vec<f64> = (-radius..=radius)
        .map(|offset| (-(offset * offset) as f64 / denom).exp())
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|value| (value / sum) as f32).collect()
}

/// `vlfeat_compatible_octave_count`: COLMAP's four octaves unless the
/// configuration says otherwise, bounded by the image.
fn detection_octaves(configured: usize, width: usize, height: usize) -> usize {
    let min_dimension = width.min(height).max(1) as f64;
    let auto = ((min_dimension.log2().floor() as i64) - 2).max(1) as usize;
    if configured == 0 {
        4usize.min(auto).max(1)
    } else {
        configured.min(auto).max(1)
    }
}

/// The descriptor pyramid's octaves down to 16 pixels (`build_vlfeat_pyramid`
/// builds at most these; a keypoint never selects past them).
fn descriptor_octaves(width: usize, height: usize) -> usize {
    let min_dimension = width.min(height).max(1) as f64;
    if min_dimension >= 16.0 {
        (min_dimension / 16.0).log2().floor() as usize + 1
    } else {
        1
    }
}

fn dispatch_linear(threads: usize, per_group: usize) -> (u32, u32, u32) {
    let groups = threads.div_ceil(per_group).max(1) as u32;
    if groups <= 65535 {
        (groups, 1, 1)
    } else {
        (65535, groups.div_ceil(65535), 1)
    }
}

fn staging_buffer(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vl-staging"),
        size: bytes.max(16).div_ceil(4) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Encode copies of `(buffer, byte offset, length)` ranges into `staging`
/// after the work in `encoder` and submit it all; the bytes copied.
fn submit_copies(
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    ranges: &[(&wgpu::Buffer, u64, u64)],
) -> (wgpu::SubmissionIndex, u64) {
    let mut at = 0u64;
    for &(buffer, offset, len) in ranges {
        if len > 0 {
            encoder.copy_buffer_to_buffer(buffer, offset, staging, at, len);
            at += len;
        }
    }
    (queue.submit(Some(encoder.finish())), at)
}

/// The first `len` bytes of `staging` once submission `index` is done (not
/// waiting for later submissions).
async fn map_read(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    index: wgpu::SubmissionIndex,
    len: u64,
) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let slice = staging.slice(..len);
    mapped(device, slice, Some(index))
        .await
        .expect("map staging");
    let data = slice.get_mapped_range().expect("map range").to_vec();
    staging.unmap();
    data
}

impl VlfeatSiftGpu {
    pub fn new(ctx: GpuContext) -> Self {
        let dev = &ctx.device;
        let module = |label: &str, src: &str| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let image = module("vl-image", include_str!("shaders/image.wgsl"));
        let detect = module("vl-detect", include_str!("shaders/vlfeat_detect.wgsl"));
        let describe = module("vl-describe", include_str!("shaders/vlfeat_describe.wgsl"));
        let mk = |m: &wgpu::ShaderModule, entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = Pipes {
            upsample: mk(&image, "upsample2x_aligned"),
            blur_h: mk(&image, "blur_h"),
            blur_v: mk(&image, "blur_v"),
            halve: mk(&image, "halve"),
            dog: mk(&detect, "vl_dog"),
            extrema: mk(&detect, "vl_extrema"),
            orient_args: mk(&detect, "vl_orient_args"),
            orient: mk(&detect, "vl_orient"),
            gradient: mk(&describe, "vl_gradient"),
            describe: mk(&describe, "vl_describe"),
        };
        Self {
            ctx,
            pipes,
            cap: INITIAL_CAP,
            plans: Vec::new(),
            pools: Vec::new(),
        }
    }

    /// Free every plan and its buffers; the next extraction allocates them
    /// again.
    pub fn clear_plan_cache(&mut self) {
        self.plans.clear();
        self.pools.clear();
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    pub fn into_context(self) -> GpuContext {
        self.ctx
    }

    /// Starts with room for `cap` candidates per octave (grown on demand).
    #[cfg(test)]
    fn with_capacity(ctx: GpuContext, cap: usize) -> Self {
        Self {
            cap,
            ..Self::new(ctx)
        }
    }

    /// Whether `config` selects the VLFeat-compatible path this extractor
    /// implements (three scales per octave, without domain-size pooling).
    pub fn supports(config: &SiftConfig) -> bool {
        config.detector == SiftDetector::Dog
            && config.intervals == 3
            && config.vlfeat_compatible_detector
            && config.vlfeat_compatible_descriptor
            && !config.domain_size_pooling
            && !config.affine
            && !config.multi_anisotropy
    }

    /// Keypoints and descriptors with the CPU `extract_sift`'s contract for a
    /// VLFeat-compatible configuration: the same order, orientation cap and
    /// per-level `max_keypoints` cap, descriptors as byte values (0..=255)
    /// in COLMAP's UBC layout.
    pub fn extract(
        &mut self,
        image: &GrayImage<'_>,
        config: &SiftConfig,
    ) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftGpuError> {
        pollster::block_on(self.extract_async(image, config))
    }

    /// [`Self::extract`] without blocking on the GPU readbacks, for a browser
    /// (WebGPU), where nothing may block.
    pub async fn extract_async(
        &mut self,
        image: &GrayImage<'_>,
        config: &SiftConfig,
    ) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftGpuError> {
        Self::check(image, config)?;
        let candidates = loop {
            self.ensure_plans(PlanKey::new(image.width, image.height, self.cap, config), 1)?;
            let index = self.submit_detect(0, image);
            match self.read_counts(0, index).await {
                Ok(counts) => {
                    let (index, len) = self.submit_candidate_copy(0, &counts);
                    break self.read_candidates(0, index, &counts, len).await;
                }
                Err(needed) => self.cap = needed.next_power_of_two(),
            }
        };
        let keypoints = select_keypoints(candidates, config);
        let descriptors = self.describe(0, &keypoints, config).await;
        Ok((keypoints, descriptors))
    }

    /// [`Self::extract`] for each image, with the same results; images of
    /// one size are pipelined: the next image is detected on the GPU while
    /// the host orders and caps the current one's candidates.
    #[allow(clippy::type_complexity)]
    pub fn extract_batch(
        &mut self,
        images: &[GrayImage<'_>],
        config: &SiftConfig,
    ) -> Result<Vec<(Vec<SiftKeypoint>, Vec<Vec<f32>>)>, SiftGpuError> {
        pollster::block_on(self.extract_batch_async(images, config))
    }

    /// [`Self::extract_batch`] without blocking on the GPU readbacks.
    #[allow(clippy::type_complexity)]
    pub async fn extract_batch_async(
        &mut self,
        images: &[GrayImage<'_>],
        config: &SiftConfig,
    ) -> Result<Vec<(Vec<SiftKeypoint>, Vec<Vec<f32>>)>, SiftGpuError> {
        let same_size = images
            .windows(2)
            .all(|w| (w[0].width, w[0].height) == (w[1].width, w[1].height));
        if images.len() < 2 || !same_size {
            let mut out = Vec::with_capacity(images.len());
            for image in images {
                out.push(self.extract_async(image, config).await?);
            }
            return Ok(out);
        }
        for image in images {
            Self::check(image, config)?;
        }
        let (w0, h0) = (images[0].width, images[0].height);
        self.ensure_plans(PlanKey::new(w0, h0, self.cap, config), 2)?;
        let mut out = Vec::with_capacity(images.len());
        let mut pending = self.submit_detect(0, &images[0]);
        for i in 0..images.len() {
            let p = i % 2;
            let counts = match self.read_counts(p, pending.clone()).await {
                Ok(counts) => counts,
                Err(needed) => {
                    // An octave overflowed: grow, and finish one at a time.
                    self.cap = needed.next_power_of_two();
                    for image in &images[i..] {
                        out.push(self.extract_async(image, config).await?);
                    }
                    return Ok(out);
                }
            };
            let (index, len) = self.submit_candidate_copy(p, &counts);
            if i + 1 < images.len() {
                pending = self.submit_detect(1 - p, &images[i + 1]);
            }
            let candidates = self.read_candidates(p, index, &counts, len).await;
            let keypoints = select_keypoints(candidates, config);
            let descriptors = self.describe(p, &keypoints, config).await;
            out.push((keypoints, descriptors));
        }
        Ok(out)
    }

    fn check(image: &GrayImage<'_>, config: &SiftConfig) -> Result<(), SiftGpuError> {
        if !Self::supports(config) {
            return Err(SiftGpuError::Unsupported);
        }
        let (w0, h0) = (image.width, image.height);
        if w0 < 16 || h0 < 16 || image.pixels.len() < w0 * h0 {
            return Err(SiftGpuError::ImageTooSmall {
                width: w0,
                height: h0,
            });
        }
        Ok(())
    }

    /// At least `count` plans for `key`, rebuilt from the kept buffers when
    /// the key changed.
    fn ensure_plans(&mut self, key: PlanKey, count: usize) -> Result<(), SiftGpuError> {
        if self.plans.first().map(|p| p.key) != Some(key) {
            self.plans.clear();
        }
        while self.plans.len() < count {
            let index = self.plans.len();
            if self.pools.len() <= index {
                self.pools.push(Pool::default());
            }
            let mut pool = std::mem::take(&mut self.pools[index]);
            let plan = self.build_plan(key, &mut pool);
            self.pools[index] = pool;
            self.plans.push(plan?);
        }
        Ok(())
    }

    fn build_plan(&self, key: PlanKey, pool: &mut Pool) -> Result<Plan, SiftGpuError> {
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let (w0, h0) = (key.width, key.height);
        let s = key.intervals;
        let levels = s + 3;
        let k = 2.0f64.powf(1.0 / s as f64);
        let cap = key.cap;

        let n_det = detection_octaves(key.octaves, w0, h0);
        let mut det_dims = Vec::with_capacity(n_det);
        let (mut w, mut h) = (2 * w0, 2 * h0);
        for _ in 0..n_det {
            det_dims.push((w, h));
            w /= 2;
            h /= 2;
        }
        let n_desc = descriptor_octaves(w0, h0);
        let mut desc_dims = Vec::with_capacity(n_desc);
        let (mut w, mut h) = (w0, h0);
        for _ in 0..n_desc {
            desc_dims.push((w, h));
            w /= 2;
            h /= 2;
        }
        // The largest binding is one doubled level, a gradient level or a
        // candidate list; past the device's limit, the CPU extractor.
        let (tw, th) = det_dims[0];
        let largest = (tw * th * 4).max(w0 * h0 * 8).max(cap * CAND_WORDS * 4) as u64;
        if largest > self.ctx.limits.max_storage_buffer_binding_size {
            return Err(SiftGpuError::Unsupported);
        }

        // Blur kernels: an identity tap (no initial blur), then the
        // detection and descriptor pyramids' initial blurs and level steps.
        let mut weights: Vec<f32> = vec![1.0];
        let identity = (0u32, 0u32);
        let mut add_kernel = |sigma: f64| -> (u32, u32) {
            let taps = gaussian_kernel_vlfeat(sigma);
            let entry = (weights.len() as u32, (taps.len() / 2) as u32);
            weights.extend_from_slice(&taps);
            entry
        };
        let step_sigmas = |base: f64| -> Vec<f64> {
            (1..levels)
                .map(|l| {
                    let target = base * k.powi(l as i32);
                    let previous = base * k.powi(l as i32 - 1);
                    (target * target - previous * previous).max(1e-8).sqrt()
                })
                .collect()
        };
        let det_base = key.sigma_base.max(2.0 * key.sigma_input).max(1e-3);
        let det_input = (2.0 * key.sigma_input).max(1e-3);
        let det_initial = if det_base > det_input {
            add_kernel((det_base * det_base - det_input * det_input).sqrt())
        } else {
            identity
        };
        let det_steps: Vec<(u32, u32)> = step_sigmas(det_base)
            .into_iter()
            .map(&mut add_kernel)
            .collect();
        let desc_base = (key.sigma_base.max(1e-3) / 2.0).max(key.sigma_input);
        let desc_input = key.sigma_input.max(1e-3);
        let desc_initial = if desc_base > desc_input {
            add_kernel((desc_base * desc_base - desc_input * desc_input).sqrt())
        } else {
            identity
        };
        let desc_steps: Vec<(u32, u32)> = step_sigmas(desc_base)
            .into_iter()
            .map(&mut add_kernel)
            .collect();

        let none = wgpu::BufferUsages::empty();
        let f32s = |n: usize| (n * 4) as u64;
        // The large buffers come from the pool; the small ones are new.
        let img = pool.storage(dev, ("vl-img", 0, 0), f32s(w0 * h0));
        let wts = storage(dev, "vl-weights", f32s(weights.len()), none);
        queue.write_buffer(&wts, 0, bytemuck::cast_slice(&weights));
        let tmp = pool.storage(dev, ("vl-tmp", 0, 0), f32s(tw * th));
        let mut levels_of =
            |label: &'static str, dims: &[(usize, usize)], count: usize, per_pixel: usize| {
                dims.iter()
                    .enumerate()
                    .map(|(o, &(w, h))| {
                        (0..count)
                            .map(|l| pool.storage(dev, (label, o, l), f32s(per_pixel * w * h)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
        let det_gauss = levels_of("vl-gauss", &det_dims, levels, 1);
        let det_dog = levels_of("vl-dog", &det_dims, levels - 1, 1);
        let desc_gauss = levels_of("vl-desc-gauss", &desc_dims, levels, 1);
        let grad = levels_of("vl-grad", &desc_dims, s, 2);
        let cand: Vec<wgpu::Buffer> = (0..n_det)
            .map(|o| pool.storage(dev, ("vl-cand", o, 0), (cap * CAND_WORDS * 4) as u64))
            .collect();
        let counters = storage(dev, "vl-counters", f32s(n_det.max(4)), none);
        let args = storage(
            dev,
            "vl-args",
            f32s(3 * n_det),
            wgpu::BufferUsages::INDIRECT,
        );
        let staging = pool.staging(
            dev,
            ("vl-staging", 0, 0),
            (n_det * cap * CAND_WORDS * 4) as u64,
        );

        let mut slots = Slots::default();
        let mut ops: Vec<Op> = Vec::new();
        let f = |v: f64| (v as f32).to_bits();
        // Detection: octave bases, Gaussian levels, DoG, extrema, orientations.
        for (o, &(w, h)) in det_dims.iter().enumerate() {
            let (wu, hu) = (w as u32, h as u32);
            let gauss = &det_gauss[o];
            if o == 0 {
                // Upsample into level 1 (scratch), blur into level 0.
                let slot = slots.push(&[wu, hu, 0, 0, 0, 0, w0 as u32, h0 as u32]);
                ops.push(Op {
                    pipe: 0,
                    slot,
                    bufs: vec![(1, &img), (2, &gauss[1])],
                    groups: groups2d(w, h),
                });
                blur_ops(
                    &mut ops,
                    &mut slots,
                    (w, h),
                    &gauss[1],
                    &tmp,
                    &gauss[0],
                    det_initial,
                );
            } else {
                let (pw, ph) = det_dims[o - 1];
                let slot = slots.push(&[wu, hu, 0, 0, 0, 0, pw as u32, ph as u32]);
                ops.push(Op {
                    pipe: 3,
                    slot,
                    bufs: vec![(1, &det_gauss[o - 1][s]), (2, &gauss[0])],
                    groups: groups2d(w, h),
                });
            }
            for (l, &kern) in det_steps.iter().enumerate() {
                blur_ops(
                    &mut ops,
                    &mut slots,
                    (w, h),
                    &gauss[l],
                    &tmp,
                    &gauss[l + 1],
                    kern,
                );
            }
            let vl = |level: usize| {
                [
                    wu,
                    hu,
                    o as u32,
                    cap as u32,
                    f(key.contrast_threshold / s as f64),
                    f(key.edge_threshold),
                    f(det_base),
                    f(k),
                    f((s + 1) as f64),
                    key.bilinear as u32,
                    level as u32,
                    0,
                ]
            };
            for l in 0..levels - 1 {
                let slot = slots.push(&vl(l));
                ops.push(Op {
                    pipe: 4,
                    slot,
                    bufs: vec![(1, &gauss[l]), (2, &gauss[l + 1]), (3, &det_dog[o][l])],
                    groups: groups2d(w, h),
                });
            }
            for di in 1..=s {
                let slot = slots.push(&vl(di));
                ops.push(Op {
                    pipe: 5,
                    slot,
                    bufs: vec![
                        (4, &det_dog[o][di - 1]),
                        (5, &det_dog[o][di]),
                        (6, &det_dog[o][di + 1]),
                        (7, &cand[o]),
                        (8, &counters),
                    ],
                    groups: groups2d(w, h),
                });
            }
            // One workgroup per candidate; the candidates of DoG level d
            // read Gaussian level d.
            let slot = slots.push(&vl(0));
            ops.push(Op {
                pipe: 8,
                slot,
                bufs: vec![(8, &counters), (12, &args)],
                groups: Dispatch::Grid(1, 1, 1),
            });
            ops.push(Op {
                pipe: 6,
                slot,
                bufs: vec![
                    (7, &cand[o]),
                    (8, &counters),
                    (9, &gauss[1]),
                    (10, &gauss[2]),
                    (11, &gauss[3]),
                ],
                groups: Dispatch::Indirect((o * 12) as u64),
            });
        }
        // Descriptor pyramid at the input resolution, and the gradients of
        // the levels a keypoint can select.
        for (o, &(w, h)) in desc_dims.iter().enumerate() {
            let (wu, hu) = (w as u32, h as u32);
            let gauss = &desc_gauss[o];
            if o == 0 {
                blur_ops(
                    &mut ops,
                    &mut slots,
                    (w, h),
                    &img,
                    &tmp,
                    &gauss[0],
                    desc_initial,
                );
            } else {
                let (pw, ph) = desc_dims[o - 1];
                let slot = slots.push(&[wu, hu, 0, 0, 0, 0, pw as u32, ph as u32]);
                ops.push(Op {
                    pipe: 3,
                    slot,
                    bufs: vec![(1, &desc_gauss[o - 1][s]), (2, &gauss[0])],
                    groups: groups2d(w, h),
                });
            }
            for (l, &kern) in desc_steps.iter().enumerate() {
                blur_ops(
                    &mut ops,
                    &mut slots,
                    (w, h),
                    &gauss[l],
                    &tmp,
                    &gauss[l + 1],
                    kern,
                );
            }
            for l in 1..=s {
                let slot = slots.push(&[wu, hu, 0, 0]);
                ops.push(Op {
                    pipe: 7,
                    slot,
                    bufs: vec![(1, &gauss[l]), (2, &grad[o][l - 1])],
                    groups: groups2d(w, h),
                });
            }
        }

        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vl-params"),
            size: (slots.words.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&slots.words));
        let pipes = self.pipes.list();
        let passes = ops
            .iter()
            .map(|op| {
                let pipe = pipes[op.pipe];
                let mut entries = vec![wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: op.slot,
                        size: NonZeroU64::new(SLOT_BIND),
                    }),
                }];
                for &(binding, buffer) in &op.bufs {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    });
                }
                if op.pipe == 1 || op.pipe == 2 {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wts.as_entire_binding(),
                    });
                }
                let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("vl-op"),
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &entries,
                });
                (op.pipe, bind, op.groups)
            })
            .collect();
        drop(ops);

        let mut levels_kept = vec![wts, tmp];
        levels_kept.extend(det_gauss.into_iter().flatten());
        levels_kept.extend(det_dog.into_iter().flatten());
        levels_kept.extend(desc_gauss.into_iter().flatten());
        Ok(Plan {
            key,
            det_dims,
            desc_dims,
            img,
            cand,
            counters,
            args,
            staging,
            grad,
            desc: None,
            _levels: levels_kept,
            _params: params,
            passes,
        })
    }

    /// Upload `image` to plan `p` and submit both scale spaces, the refined
    /// extrema, their orientations and the counters' copy.
    fn submit_detect(&self, p: usize, image: &GrayImage<'_>) -> wgpu::SubmissionIndex {
        let plan = &self.plans[p];
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let (w0, h0) = (image.width, image.height);
        queue.write_buffer(&plan.img, 0, bytemuck::cast_slice(&image.pixels[..w0 * h0]));
        let n_det = plan.det_dims.len();
        queue.write_buffer(&plan.counters, 0, &vec![0u8; n_det.max(4) * 4]);
        let pipes = self.pipes.list();
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vl-detect"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vl-detect"),
                timestamp_writes: None,
            });
            for (pipe, bind, groups) in &plan.passes {
                pass.set_pipeline(pipes[*pipe]);
                pass.set_bind_group(0, bind, &[]);
                match *groups {
                    Dispatch::Grid(x, y, z) => pass.dispatch_workgroups(x, y, z),
                    Dispatch::Indirect(offset) => {
                        pass.dispatch_workgroups_indirect(&plan.args, offset)
                    }
                }
            }
        }
        let counters = [(&plan.counters, 0, (n_det * 4) as u64)];
        submit_copies(queue, encoder, &plan.staging, &counters).0
    }

    /// Plan `p`'s candidate counts per octave, `Err(needed)` when one
    /// overflowed the capacity.
    async fn read_counts(&self, p: usize, index: wgpu::SubmissionIndex) -> Result<Vec<u32>, usize> {
        let plan = &self.plans[p];
        let n_det = plan.det_dims.len();
        let bytes = map_read(&self.ctx.device, &plan.staging, index, (n_det * 4) as u64).await;
        let counts = u32_words(&bytes);
        match counts.iter().max() {
            Some(&max) if max as usize > plan.key.cap => Err(max as usize),
            _ => Ok(counts),
        }
    }

    /// Submit the copy of plan `p`'s candidates; the bytes copied.
    fn submit_candidate_copy(&self, p: usize, counts: &[u32]) -> (wgpu::SubmissionIndex, u64) {
        let plan = &self.plans[p];
        let ranges: Vec<(&wgpu::Buffer, u64, u64)> = counts
            .iter()
            .zip(&plan.cand)
            .map(|(&c, buffer)| (buffer, 0, (c as usize * CAND_WORDS * 4) as u64))
            .collect();
        let encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vl-readback"),
            });
        submit_copies(&self.ctx.queue, encoder, &plan.staging, &ranges)
    }

    /// Plan `p`'s candidates, octave by octave.
    async fn read_candidates(
        &self,
        p: usize,
        index: wgpu::SubmissionIndex,
        counts: &[u32],
        len: u64,
    ) -> Vec<Candidate> {
        let bytes = map_read(&self.ctx.device, &self.plans[p].staging, index, len).await;
        let raw = u32_words(&bytes);
        let mut candidates = Vec::with_capacity(raw.len() / CAND_WORDS);
        let mut chunks = raw.chunks_exact(CAND_WORDS);
        for (o, &c) in counts.iter().enumerate() {
            for r in chunks.by_ref().take(c as usize) {
                candidates.push(Candidate {
                    octave: o,
                    ox: r[0],
                    oy: r[1],
                    level: r[2],
                    x: f32::from_bits(r[4]),
                    y: f32::from_bits(r[5]),
                    scale: f32::from_bits(r[6]),
                    orientations: (r[3] as usize).min(4),
                    angles: [
                        f32::from_bits(r[9]),
                        f32::from_bits(r[10]),
                        f32::from_bits(r[11]),
                        f32::from_bits(r[12]),
                    ],
                });
            }
        }
        candidates
    }

    /// Descriptors of `keypoints`, one dispatch per gradient level with that
    /// level's keypoints.
    async fn describe(
        &mut self,
        p: usize,
        keypoints: &[SiftKeypoint],
        config: &SiftConfig,
    ) -> Vec<Vec<f32>> {
        if keypoints.is_empty() {
            return Vec::new();
        }
        let plan = &self.plans[p];
        let s = plan.key.intervals;
        let k = 2.0f64.powf(1.0 / s as f64);
        let base = (plan.key.sigma_base.max(1e-3) / 2.0).max(plan.key.sigma_input);
        let n_desc = plan.desc_dims.len();
        let jobs: Vec<DescKp> = keypoints
            .iter()
            .map(|kp| descriptor_job(kp, base, k, s + 3, &plan.desc_dims))
            .collect();
        // Keypoints grouped by gradient level, each group in input order.
        let group = |j: &DescKp| j.octave * s + j.level - 1;
        let n_groups = n_desc * s;
        let mut order: Vec<usize> = (0..jobs.len()).collect();
        order.sort_by_key(|&i| group(&jobs[i]));
        let mut ranges = vec![(0u32, 0u32); n_groups];
        for (at, &i) in order.iter().enumerate() {
            let g = group(&jobs[i]);
            if ranges[g].1 == 0 {
                ranges[g].0 = at as u32;
            }
            ranges[g].1 += 1;
        }
        let words: Vec<u32> = order.iter().flat_map(|&i| jobs[i].words()).collect();

        let n = keypoints.len();
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        if plan.desc.as_ref().is_none_or(|d| d.capacity < n) {
            let capacity = n.next_power_of_two().max(1024);
            let params = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vl-desc-params"),
                size: SLOT * n_groups as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pool = &mut self.pools[p];
            let kps = pool.storage(dev, ("vl-desc-kps", 0, 0), (capacity * KP_WORDS * 4) as u64);
            let out = pool.storage(dev, ("vl-desc-out", 0, 0), (capacity * 32 * 4) as u64);
            let staging = pool.staging(dev, ("vl-desc-staging", 0, 0), (capacity * 128) as u64);
            let binds = (0..n_groups)
                .map(|g| {
                    dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vl-describe"),
                        layout: &self.pipes.describe.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params,
                                    offset: SLOT * g as u64,
                                    size: NonZeroU64::new(16),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: plan.grad[g / s][g % s].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: kps.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: out.as_entire_binding(),
                            },
                        ],
                    })
                })
                .collect();
            self.plans[p].desc = Some(DescBuffers {
                capacity,
                params,
                kps,
                out,
                staging,
                binds,
            });
        }
        let desc = self.plans[p].desc.as_ref().expect("descriptor buffers");
        let l1 = matches!(config.normalization, SiftNormalization::L1Root) as u32;
        let mut slots = Slots::default();
        for &(first, num) in &ranges {
            slots.push(&[first, num, l1, 0]);
        }
        queue.write_buffer(&desc.params, 0, bytemuck::cast_slice(&slots.words));
        queue.write_buffer(&desc.kps, 0, bytemuck::cast_slice(&words));
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vl-describe"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vl-describe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipes.describe);
            for (g, &(_, num)) in ranges.iter().enumerate() {
                if num == 0 {
                    continue;
                }
                pass.set_bind_group(0, &desc.binds[g], &[]);
                // One workgroup per keypoint.
                let groups = dispatch_linear(num as usize, 1);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        let (index, len) = submit_copies(
            queue,
            encoder,
            &desc.staging,
            &[(&desc.out, 0, (n * 128) as u64)],
        );
        let bytes = map_read(dev, &desc.staging, index, len).await;
        let mut descriptors = vec![Vec::new(); n];
        for (at, &i) in order.iter().enumerate() {
            descriptors[i] = bytes[at * 128..(at + 1) * 128]
                .iter()
                .map(|&b| b as f32)
                .collect();
        }
        descriptors
    }
}

/// `VlfeatPyramid::select` and `describe_raw_vlfeat`'s window, in f64: the
/// gradient level, its sampling window and the keypoint frame.
fn descriptor_job(
    kp: &SiftKeypoint,
    base: f64,
    k: f64,
    levels: usize,
    desc_dims: &[(usize, usize)],
) -> DescKp {
    let sigma = if kp.sigma.is_finite() && kp.sigma > 0.0 {
        kp.sigma
    } else {
        base
    };
    let ratio = (sigma / base).max(f64::MIN_POSITIVE);
    let intervals = k.log2().recip();
    let octave_phase = ratio.log2() - 0.5 / intervals;
    let octave = (octave_phase.floor().max(0.0) as usize).min(desc_dims.len() - 1);
    let pixel_scale = 2.0f64.powi(octave as i32);
    let local_sigma = sigma / pixel_scale;
    let top = levels.saturating_sub(3) as f64;
    let position = ((local_sigma / base).max(f64::MIN_POSITIVE).log2() * intervals).clamp(1.0, top);
    let level = position.round().clamp(1.0, top) as usize;
    let sbp = (3.0 * local_sigma + f64::EPSILON).max(1e-6);
    let (lw, lh) = desc_dims[octave];
    let local_x = (kp.x - 0.5) / pixel_scale;
    let local_y = (kp.y - 0.5) / pixel_scale;
    let xi = (local_x + 0.5).floor() as i64;
    let yi = (local_y + 0.5).floor() as i64;
    let window = if sbp.is_finite() && sbp > 0.0 {
        (std::f64::consts::SQRT_2 * sbp * 5.0 / 2.0 + 0.5).floor() as i64
    } else {
        0
    };
    let (mut min_dx, mut max_dx) = ((-window).max(1 - xi), window.min(lw as i64 - 2 - xi));
    let (mut min_dy, mut max_dy) = ((-window).max(1 - yi), window.min(lh as i64 - 2 - yi));
    if min_dx > max_dx || min_dy > max_dy {
        (min_dx, max_dx, min_dy, max_dy) = (0, -1, 0, -1);
    }
    DescKp {
        octave,
        level,
        lw: lw as u32,
        xi: xi as i32,
        yi: yi as i32,
        min_dx: min_dx as i32,
        max_dx: max_dx as i32,
        min_dy: min_dy as i32,
        max_dy: max_dy as i32,
        off_x: (xi as f64 - local_x) as f32,
        off_y: (yi as f64 - local_y) as f32,
        sbp: sbp as f32,
        cos_o: kp.orientation.cos() as f32,
        sin_o: kp.orientation.sin() as f32,
        orientation: kp.orientation as f32,
    }
}

/// The CPU detector's bookkeeping on the GPU's candidates: its scan order
/// (octave, level, row, column), the per-locus orientation cap, then
/// `cap_vlfeat_oriented_detections_by_level` (a suffix of whole
/// (octave, level) groups, counted in unoriented extrema).
fn select_keypoints(mut candidates: Vec<Candidate>, config: &SiftConfig) -> Vec<SiftKeypoint> {
    candidates.sort_unstable_by_key(|c| (c.octave, c.level, c.oy, c.ox));
    let s = config.intervals.max(1);
    let k = 2.0f64.powf(1.0 / s as f64);
    let base = config.sigma_base.max(2.0 * config.sigma_input).max(1e-3);
    let orientation_cap = if config.max_orientations == 0 {
        2
    } else {
        config.max_orientations.min(4)
    };
    // Groups in scan order and their candidate counts.
    let mut group_of = Vec::with_capacity(candidates.len());
    let mut groups: Vec<usize> = Vec::new();
    let mut last = None;
    for c in &candidates {
        let key = (c.octave, c.level);
        if last != Some(key) {
            groups.push(0);
            last = Some(key);
        }
        *groups.last_mut().expect("group") += 1;
        group_of.push(groups.len() - 1);
    }
    let mut first_group = 0;
    if config.max_keypoints != usize::MAX {
        let mut counted = 0usize;
        for g in (0..groups.len()).rev() {
            counted = counted.saturating_add(groups[g]);
            if counted > config.max_keypoints {
                first_group = g;
                break;
            }
        }
    }
    let mut keypoints = Vec::new();
    for (c, &g) in candidates.iter().zip(&group_of) {
        if g < first_group {
            continue;
        }
        let xper = 2.0f64.powi(c.octave as i32) / 2.0;
        let local_sigma = base * k.powf(c.scale as f64 + 1.0);
        for &angle in c.angles.iter().take(c.orientations.min(orientation_cap)) {
            keypoints.push(SiftKeypoint::from_location_scale_orientation(
                c.x as f64 * xper + 0.5,
                c.y as f64 * xper + 0.5,
                local_sigma * xper,
                angle as f64,
            ));
        }
    }
    keypoints
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_context;
    use visloc_vision::features::sift::extract_sift;

    /// `gsplat_photos`' SIFT settings.
    fn photos_config() -> SiftConfig {
        SiftConfig {
            max_keypoints: 8192,
            normalization: SiftNormalization::L1Root,
            descriptor_magnification: 3.0,
            max_orientations: 2,
            prefer_larger_scale: true,
            vlfeat_compatible_detector: true,
            vlfeat_compatible_descriptor: true,
            vlfeat_bilinear_orientations: true,
            ..SiftConfig::default()
        }
    }

    /// Deterministic texture in [0, 1]: blobs of several sizes over smooth
    /// waves, so every octave has extrema.
    fn texture(w: usize, h: usize) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let blobs: Vec<(f64, f64, f64, f64)> = (0..90)
            .map(|_| {
                (
                    next() * w as f64,
                    next() * h as f64,
                    1.0 + 9.0 * next() * next(),
                    if next() < 0.5 { -0.35 } else { 0.35 },
                )
            })
            .collect();
        let mut px = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                let mut v = 0.5 + 0.08 * (fx * 0.11).sin() * (fy * 0.07).cos();
                for &(cx, cy, s, a) in &blobs {
                    let d2 = (fx - cx).powi(2) + (fy - cy).powi(2);
                    v += a * (-d2 / (2.0 * s * s)).exp();
                }
                px[y * w + x] = v.clamp(0.0, 1.0) as f32;
            }
        }
        px
    }

    /// A featureless image reads back no candidates: an empty readback,
    /// which `bytemuck::cast_slice` rejected as misaligned (panic).
    #[test]
    fn featureless_images_have_no_features() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let flat = vec![0.5f32; 160 * 112];
        let image = GrayImage::new(160, 112, &flat).unwrap();
        let mut gpu = VlfeatSiftGpu::new(ctx);
        let (keypoints, descriptors) = gpu.extract(&image, &photos_config()).unwrap();
        assert!(keypoints.is_empty() && descriptors.is_empty());
    }

    /// Keypoints of `a` with a counterpart in `b` (position within 0.01 px,
    /// scale within 0.1%, orientation within 0.01 rad), as index pairs.
    fn pair_up(a: &[SiftKeypoint], b: &[SiftKeypoint]) -> Vec<(usize, usize)> {
        let mut used = vec![false; b.len()];
        let mut pairs = Vec::new();
        for (i, p) in a.iter().enumerate() {
            let hit = b.iter().enumerate().position(|(j, q)| {
                let d = (p.orientation - q.orientation).rem_euclid(std::f64::consts::TAU);
                !used[j]
                    && (p.x - q.x).abs() < 0.01
                    && (p.y - q.y).abs() < 0.01
                    && (p.sigma - q.sigma).abs() < 1e-3 * p.sigma
                    && d.min(std::f64::consts::TAU - d) < 0.01
            });
            if let Some(j) = hit {
                used[j] = true;
                pairs.push((i, j));
            }
        }
        pairs
    }

    #[test]
    fn gpu_vlfeat_sift_matches_the_cpu_path() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (w, h) = (256, 176);
        let px = texture(w, h);
        let image = GrayImage::new(w, h, &px).unwrap();
        let config = photos_config();
        let (ck, cd) = extract_sift(&image, &config).unwrap();
        let mut gpu = VlfeatSiftGpu::new(ctx);
        let (gk, gd) = gpu.extract(&image, &config).unwrap();
        assert!(ck.len() > 100, "{} CPU keypoints", ck.len());
        let pairs = pair_up(&ck, &gk);
        let found = pairs.len() as f64 / ck.len() as f64;
        let precise = pairs.len() as f64 / gk.len() as f64;
        eprintln!(
            "cpu {} gpu {} paired {} ({:.4} of cpu, {:.4} of gpu)",
            ck.len(),
            gk.len(),
            pairs.len(),
            found,
            precise
        );
        assert!(found >= 0.98 && precise >= 0.98);
        // Paired keypoints are in the same order.
        assert!(pairs.windows(2).all(|p| p[0].1 < p[1].1));
        let mut worst = 0.0f32;
        let mut sum = 0.0f64;
        for &(i, j) in &pairs {
            for (a, b) in cd[i].iter().zip(&gd[j]) {
                worst = worst.max((a - b).abs());
                sum += (a - b).abs() as f64;
            }
        }
        let mean = sum / (pairs.len() * 128) as f64;
        eprintln!("descriptor bytes: worst {worst}, mean {mean:.5}");
        assert!(worst <= 3.0 && mean < 0.02, "worst {worst} mean {mean}");
        // A second image of the same size reuses the plan.
        let (gk2, gd2) = gpu.extract(&image, &config).unwrap();
        assert_eq!(gk, gk2);
        assert_eq!(gd, gd2);
    }

    /// An octave with more candidates than the buffer holds is redone with
    /// a larger one, and the result does not depend on the starting size.
    #[test]
    fn candidate_overflow_grows_the_buffer() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (w, h) = (160, 112);
        let px = texture(w, h);
        let image = GrayImage::new(w, h, &px).unwrap();
        let config = photos_config();
        let mut small = VlfeatSiftGpu::with_capacity(ctx, 4);
        let (sk, sd) = small.extract(&image, &config).unwrap();
        assert!(small.cap > 4, "capacity grew to {}", small.cap);
        let mut big = VlfeatSiftGpu::new(small.into_context());
        let (bk, bd) = big.extract(&image, &config).unwrap();
        assert!(!sk.is_empty());
        assert_eq!(sk, bk);
        assert_eq!(sd, bd);
    }

    /// Alternating image sizes give what a fresh extractor gives, and only
    /// rebuild the plan's dispatches.
    #[test]
    fn image_sizes_can_alternate() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let config = photos_config();
        let small = texture(160, 112);
        let large = texture(256, 176);
        let small_image = GrayImage::new(160, 112, &small).unwrap();
        let large_image = GrayImage::new(256, 176, &large).unwrap();
        let mut gpu = VlfeatSiftGpu::new(ctx);
        let first_small = gpu.extract(&small_image, &config).unwrap();
        let first_large = gpu.extract(&large_image, &config).unwrap();
        let second_small = gpu.extract(&small_image, &config).unwrap();
        let second_large = gpu.extract(&large_image, &config).unwrap();
        assert_eq!(first_small, second_small);
        assert_eq!(first_large, second_large);
        // The second round reuses the first round's buffers.
        let allocated = gpu.pools[0].allocated.len();
        assert_eq!(gpu.extract(&small_image, &config).unwrap(), first_small);
        assert_eq!(gpu.extract(&large_image, &config).unwrap(), first_large);
        assert_eq!(gpu.pools[0].allocated.len(), allocated);
        let mut fresh = VlfeatSiftGpu::new(gpu.into_context());
        assert_eq!(fresh.extract(&large_image, &config).unwrap(), first_large);
    }

    /// A smaller image after a larger one allocates nothing; a grown
    /// candidate capacity only reallocates the candidate buffers and their
    /// staging copy; a cleared cache allocates again and extracts the same.
    #[test]
    fn buffers_are_kept_for_the_largest_image() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let config = photos_config();
        let small = texture(160, 112);
        let large = texture(256, 176);
        let small_image = GrayImage::new(160, 112, &small).unwrap();
        let large_image = GrayImage::new(256, 176, &large).unwrap();
        let mut gpu = VlfeatSiftGpu::new(ctx);
        let first_large = gpu.extract(&large_image, &config).unwrap();
        let allocated = gpu.pools[0].allocated.len();
        let first_small = gpu.extract(&small_image, &config).unwrap();
        assert_eq!(gpu.pools[0].allocated.len(), allocated);

        // A featureless image keeps the small capacity; the textured one of
        // the same size outgrows it.
        let flat = vec![0.5f32; 256 * 176];
        let flat_image = GrayImage::new(256, 176, &flat).unwrap();
        let mut grown = VlfeatSiftGpu::with_capacity(gpu.into_context(), 4);
        grown.extract(&flat_image, &config).unwrap();
        let allocated = grown.pools[0].allocated.len();
        assert_eq!(grown.extract(&large_image, &config).unwrap(), first_large);
        assert!(grown.cap > 4, "capacity grew to {}", grown.cap);
        let reallocated = &grown.pools[0].allocated[allocated..];
        let n_det = grown.plans[0].det_dims.len();
        // The grown candidate lists and their staging copy, then the
        // descriptor buffers the featureless image never needed.
        assert_eq!(reallocated.len(), n_det + 1 + 3, "{reallocated:?}");
        assert!(reallocated.iter().all(|(label, _, _)| {
            [
                "vl-cand",
                "vl-staging",
                "vl-desc-kps",
                "vl-desc-out",
                "vl-desc-staging",
            ]
            .contains(label)
        }));

        grown.clear_plan_cache();
        assert!(grown.plans.is_empty() && grown.pools.is_empty());
        assert_eq!(grown.extract(&large_image, &config).unwrap(), first_large);
        assert_eq!(grown.extract(&small_image, &config).unwrap(), first_small);
    }

    /// The pipelined batch gives each image what `extract` gives it, also
    /// across an overflow that grows the candidate buffer midway.
    #[test]
    fn batch_matches_one_at_a_time() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let config = photos_config();
        let (w, h) = (192, 128);
        let pixels: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let mut px = texture(w + 8 * i, h);
                px.truncate(w * h);
                px
            })
            .collect();
        let images: Vec<GrayImage> = pixels
            .iter()
            .map(|px| GrayImage::new(w, h, px).unwrap())
            .collect();
        let mut one = VlfeatSiftGpu::new(ctx);
        let expected: Vec<_> = images
            .iter()
            .map(|im| one.extract(im, &config).unwrap())
            .collect();
        let mut batch = VlfeatSiftGpu::new(one.into_context());
        assert_eq!(batch.extract_batch(&images, &config).unwrap(), expected);
        let mut small = VlfeatSiftGpu::with_capacity(batch.into_context(), 4);
        assert_eq!(small.extract_batch(&images, &config).unwrap(), expected);
        assert!(small.cap > 4);
    }

    /// The per-level feature cap keeps a suffix of whole (octave, level)
    /// groups, counted in unoriented extrema, like the CPU path.
    #[test]
    fn feature_cap_matches_the_cpu_path() {
        let Some(ctx) = try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (w, h) = (256, 176);
        let px = texture(w, h);
        let image = GrayImage::new(w, h, &px).unwrap();
        let mut gpu = VlfeatSiftGpu::new(ctx);
        for max_keypoints in [10, 40, 70] {
            let config = SiftConfig {
                max_keypoints,
                ..photos_config()
            };
            let (ck, _) = extract_sift(&image, &config).unwrap();
            let (gk, _) = gpu.extract(&image, &config).unwrap();
            let pairs = pair_up(&ck, &gk);
            assert!(
                pairs.len() + 1 >= ck.len() && pairs.len() + 1 >= gk.len(),
                "max {max_keypoints}: cpu {} gpu {} paired {}",
                ck.len(),
                gk.len(),
                pairs.len()
            );
        }
    }
}
