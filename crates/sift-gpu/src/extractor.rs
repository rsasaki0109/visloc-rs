//! Host side of the GPU SIFT extractor.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use thiserror::Error;
use visloc_gsplat_render::GpuContext;
use visloc_vision::features::sift::{
    GrayImage, SiftConfig, SiftDetector, SiftKeypoint, SiftNormalization,
};

async fn map_buffer_async(
    device: &wgpu::Device,
    slice: wgpu::BufferSlice<'_>,
) -> Result<(), wgpu::BufferAsyncError> {
    let completion: Arc<
        Mutex<(
            Option<Result<(), wgpu::BufferAsyncError>>,
            Option<std::task::Waker>,
        )>,
    > = Arc::new(Mutex::new((None, None)));
    let callback_state = Arc::clone(&completion);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let mut state = callback_state.lock().expect("map callback lock");
        state.0 = Some(result);
        if let Some(waker) = state.1.take() {
            waker.wake();
        }
    });

    // wgpu invokes map callbacks while the device is polled. Drive that poll
    // on a helper thread so the async executor thread can remain unblocked.
    let poll_device = device.clone();

    std::thread::spawn(move || {
        let _ = poll_device.poll(wgpu::PollType::wait_indefinitely());
    });

    std::future::poll_fn(|cx| {
        {
            let mut state = completion.lock().expect("map future lock");
            if let Some(result) = state.0.take() {
                return std::task::Poll::Ready(result);
            }
            state.1 = Some(cx.waker().clone());
        }
        let mut state = completion.lock().expect("map future lock");
        if let Some(result) = state.0.take() {
            std::task::Poll::Ready(result)
        } else {
            std::task::Poll::Pending
        }
    })
    .await
}

/// Errors from the GPU extractor.
#[derive(Debug, Error)]
pub enum SiftGpuError {
    #[error("SIFT configuration is not supported on the GPU path (use the CPU extractor)")]
    Unsupported,
    #[error("image too small for SIFT: {width}x{height}")]
    ImageTooSmall { width: usize, height: usize },
}

const MAX_OCTAVES: usize = 16;
const KP_COUNTER: usize = 16;
const COUNTER_WORDS: usize = 20;
/// Uniform slot stride (>= `min_uniform_buffer_offset_alignment` everywhere).
const SLOT: u64 = 256;
const SLOT_BIND: u64 = 64;
/// Words per oriented keypoint written by `orientation`.
const KP_WORDS: usize = 9;
/// Words per raw extrema candidate written by `extrema`.
const CAND_WORDS: usize = 7;
const CAND_BYTES: usize = CAND_WORDS * 4;
/// Uniform slot size in 32-bit words.
const SLOT_WORDS: usize = (SLOT / 4) as usize;

struct Pipelines {
    upsample2x: wgpu::ComputePipeline,
    upsample2x_aligned: wgpu::ComputePipeline,
    blur_h: wgpu::ComputePipeline,
    blur_v: wgpu::ComputePipeline,
    blur_v_dog: wgpu::ComputePipeline,
    halve: wgpu::ComputePipeline,
    dog: wgpu::ComputePipeline,
    extrema: wgpu::ComputePipeline,
    orientation_args: wgpu::ComputePipeline,
    orientation: wgpu::ComputePipeline,
    describe: wgpu::ComputePipeline,
}

struct TimestampProfiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
}

#[derive(Default)]
struct BufferPool {
    ids: HashMap<String, usize>,
    entries: Vec<Option<(Arc<wgpu::Buffer>, u64, wgpu::BufferUsages)>>,
}

impl BufferPool {
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> usize {
        let needed = bytes.max(16).div_ceil(4) * 4;
        let id = if let Some(&id) = self.ids.get(label) {
            id
        } else {
            let id = self.entries.len();
            self.entries.push(None);
            self.ids.insert(label.to_owned(), id);
            id
        };
        let replace = self.entries[id]
            .as_ref()
            .is_none_or(|(_, size, old_usage)| *size < needed || !old_usage.contains(usage));
        if replace {
            let size = self.entries[id]
                .as_ref()
                .map_or(needed, |(_, size, _)| needed.max(size.saturating_mul(2)));
            self.entries[id] = Some((
                Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })),
                size,
                usage,
            ));
        }
        id
    }

    fn handle(&self, id: usize) -> Arc<wgpu::Buffer> {
        Arc::clone(
            &self.entries[id]
                .as_ref()
                .expect("pooled buffer allocated")
                .0,
        )
    }
}

/// GPU elapsed times for the scale-space, extrema, orientation, and descriptor passes.
#[derive(Debug, Clone, Copy)]
pub struct SiftGpuTimings {
    pub scale_space_ms: f64,
    pub extrema_ms: f64,
    pub orientation_ms: f64,
    pub descriptor_ms: Option<f64>,
}

/// GPU SIFT extractor (owns its [`GpuContext`]).
pub struct SiftGpu {
    ctx: GpuContext,
    pipes: Pipelines,
    buffers: BufferPool,
    timestamps: Option<TimestampProfiler>,
    /// Reusable small staging buffer for detector counters.
    counter_staging: Option<wgpu::Buffer>,
    /// Grow-on-demand staging buffer for the active extrema-candidate prefix.
    candidate_staging: Option<wgpu::Buffer>,
    /// Grow-on-demand staging buffer for the active oriented-keypoint prefix.
    keypoint_staging: Option<wgpu::Buffer>,
    /// Candidate capacity per octave as a fraction of its pixel count
    /// (grown automatically when an image overflows it).
    cand_div: usize,
}

/// One detected, oriented keypoint before the host-side ordering/truncation.
#[derive(Clone, Copy)]
struct RawKp {
    octave: u32,
    level: u32,
    x: u32,
    y: u32,
    bin: u32,
    orientation: f32,
    contrast: f32,
    dx: f32,
    dy: f32,
    ds: f32,
}

/// One raw DoG extremum as written by `extrema` (without the compact GPU
/// record's octave tag).
#[derive(Clone, Copy)]
struct CandidateRec {
    octave: u32,
    level: u32,
    x: u32,
    y: u32,
    value: f32,
    dx: f32,
    dy: f32,
    ds: f32,
}

/// Exact 28-byte layout of `Candidate` in `detect.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCandidate {
    x: u32,
    y: u32,
    level: u32,
    value: f32,
    dx: f32,
    dy: f32,
    ds: f32,
}

fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    // Same taps as visloc-vision's legacy `gaussian_kernel` (f64, then f32).
    let radius = (sigma * 3.0).ceil().max(1.0) as i64;
    let denom = 2.0 * sigma * sigma;
    let raw: Vec<f64> = (-radius..=radius)
        .map(|k| (-(k * k) as f64 / denom).exp())
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|v| (v / sum) as f32).collect()
}

pub(crate) fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sift-staging"),
        size: len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sift-staging-copy"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len as u64);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let _ = rx.recv();
    let data = slice.get_mapped_range().expect("map range").to_vec();
    staging.unmap();
    data
}

pub(crate) fn storage(
    dev: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).div_ceil(4) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | extra,
        mapped_at_creation: false,
    })
}

fn storage_usage(extra: wgpu::BufferUsages) -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC
        | extra
}

/// Uniform parameter slots written once per extraction.
struct Params {
    words: Vec<u32>,
}

impl Params {
    fn push(&mut self, fields: &[u32]) -> u64 {
        let slot = (self.words.len() * 4) as u64;
        let mut s = [0u32; (SLOT / 4) as usize];
        s[..fields.len()].copy_from_slice(fields);
        self.words.extend_from_slice(&s);
        slot
    }
}

fn dispatch_2d(n: u32) -> (u32, u32) {
    if n <= 65535 {
        (n, 1)
    } else {
        (65535, n.div_ceil(65535))
    }
}

impl SiftGpu {
    pub fn new(ctx: GpuContext) -> Self {
        Self::with_gpu_timestamps(ctx, false)
    }

    /// Create an extractor with optional pass-boundary GPU timestamp queries.
    /// Timestamp queries are enabled only when supported by the active adapter.
    pub fn with_gpu_timestamps(ctx: GpuContext, enabled: bool) -> Self {
        let dev = &ctx.device;
        let module = |label: &str, src: &str| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let image = module("sift-image", include_str!("shaders/image.wgsl"));
        let detect = module("sift-detect", include_str!("shaders/detect.wgsl"));
        let describe = module("sift-describe", include_str!("shaders/describe.wgsl"));
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
        let pipes = Pipelines {
            upsample2x: mk(&image, "upsample2x"),
            upsample2x_aligned: mk(&image, "upsample2x_aligned"),
            blur_h: mk(&image, "blur_h"),
            blur_v: mk(&image, "blur_v"),
            blur_v_dog: mk(&image, "blur_v_dog"),
            halve: mk(&image, "halve"),
            dog: mk(&image, "dog"),
            extrema: mk(&detect, "extrema"),
            orientation_args: mk(&detect, "orientation_args"),
            orientation: mk(&detect, "orientation"),
            describe: mk(&describe, "describe"),
        };
        let timestamps = if enabled && ctx.features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(TimestampProfiler {
                queries: dev.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("sift-stage-timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 8,
                }),
                resolve: dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sift-timestamp-resolve"),
                    size: 64,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                staging: dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sift-timestamp-staging"),
                    size: 64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            })
        } else {
            None
        };
        Self {
            ctx,
            pipes,
            buffers: BufferPool::default(),
            timestamps,
            counter_staging: None,
            candidate_staging: None,
            keypoint_staging: None,
            cand_div: 32,
        }
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    pub fn into_context(self) -> GpuContext {
        self.ctx
    }

    /// Approximate sum of capacities retained by the high-water extraction
    /// buffer pool. This excludes wgpu/driver overhead and the dedicated
    /// readback buffers; it is primarily useful for queue-depth budgeting.
    pub fn pooled_buffer_bytes(&self) -> u64 {
        self.buffers
            .entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|(_, size, _)| *size))
            .sum()
    }

    /// Resolve the most recent extraction's pass timestamps, if enabled and supported.
    /// Call after extraction; this waits for the tiny query readback.
    pub fn take_gpu_timings(&self, descriptor_ran: bool) -> Option<SiftGpuTimings> {
        let profiler = self.timestamps.as_ref()?;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let count = if descriptor_ran { 8 } else { 6 };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sift-timestamp-resolve"),
        });
        encoder.resolve_query_set(&profiler.queries, 0..count, &profiler.resolve, 0);
        encoder.copy_buffer_to_buffer(&profiler.resolve, 0, &profiler.staging, 0, count as u64 * 8);
        queue.submit(Some(encoder.finish()));
        let slice = profiler.staging.slice(..count as u64 * 8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().ok()?.ok()?;
        let mapped = slice.get_mapped_range().ok()?;
        let ticks: &[u64] = bytemuck::cast_slice(&mapped);
        let period_ms = queue.get_timestamp_period() as f64 / 1_000_000.0;
        let scale_space_ms = ticks[1].wrapping_sub(ticks[0]) as f64 * period_ms;
        let extrema_ms = ticks[3].wrapping_sub(ticks[2]) as f64 * period_ms;
        let orientation_ms = ticks[5].wrapping_sub(ticks[4]) as f64 * period_ms;
        let descriptor_ms =
            (count == 8).then(|| ticks[7].wrapping_sub(ticks[6]) as f64 * period_ms);
        drop(mapped);
        profiler.staging.unmap();
        Some(SiftGpuTimings {
            scale_space_ms,
            extrema_ms,
            orientation_ms,
            descriptor_ms,
        })
    }

    /// Read the counters copied by the most recent submit.
    async fn map_counter_staging(&self) -> [u32; COUNTER_WORDS] {
        let bytes = COUNTER_WORDS * 4;
        let staging = self
            .counter_staging
            .as_ref()
            .expect("counter staging allocated");
        let slice = staging.slice(..bytes as u64);
        map_buffer_async(&self.ctx.device, slice)
            .await
            .expect("map counter staging");
        let mapped = slice.get_mapped_range().expect("map detection counters");
        let mut counts = [0u32; COUNTER_WORDS];
        for (count, word) in counts.iter_mut().zip(mapped.chunks_exact(4)) {
            *count = u32::from_ne_bytes([word[0], word[1], word[2], word[3]]);
        }
        drop(mapped);
        staging.unmap();
        counts
    }

    /// Copy the active candidate prefix of each octave into one staging
    /// buffer and parse it. Used to cap candidates before orientation.
    async fn read_candidates(
        &mut self,
        cands: &[Arc<wgpu::Buffer>],
        counts: &[u32],
        octaves: usize,
    ) -> Vec<CandidateRec> {
        let total: usize = counts[..octaves].iter().map(|&count| count as usize).sum();
        if total == 0 {
            return Vec::new();
        }
        let bytes = total * CAND_BYTES;
        if self
            .candidate_staging
            .as_ref()
            .is_none_or(|staging| staging.size() < bytes as u64)
        {
            self.candidate_staging = Some(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sift-candidate-staging"),
                size: bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let staging = self
            .candidate_staging
            .as_ref()
            .expect("candidate staging allocated");
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sift-candidate-readback"),
            });
        let mut offset = 0u64;
        for (octave, &count) in counts[..octaves].iter().enumerate() {
            let len = count as u64 * CAND_BYTES as u64;
            if len > 0 {
                encoder.copy_buffer_to_buffer(&cands[octave], 0, staging, offset, len);
                offset += len;
            }
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..bytes as u64);
        map_buffer_async(&self.ctx.device, slice)
            .await
            .expect("map candidate staging");
        let mapped = slice.get_mapped_range().expect("map candidates");
        let mut candidates = Vec::with_capacity(total);
        let mut offset = 0usize;
        for (octave, &count) in counts[..octaves].iter().enumerate() {
            for _ in 0..count as usize {
                let record = &mapped[offset..offset + CAND_BYTES];
                let word = |index: usize| {
                    let start = index * 4;
                    u32::from_ne_bytes([
                        record[start],
                        record[start + 1],
                        record[start + 2],
                        record[start + 3],
                    ])
                };
                candidates.push(CandidateRec {
                    octave: octave as u32,
                    x: word(0),
                    y: word(1),
                    level: word(2),
                    value: f32::from_bits(word(3)),
                    dx: f32::from_bits(word(4)),
                    dy: f32::from_bits(word(5)),
                    ds: f32::from_bits(word(6)),
                });
                offset += CAND_BYTES;
            }
        }
        drop(mapped);
        staging.unmap();
        candidates
    }

    /// Copy the active oriented-keypoint prefix and parse it on the host.
    async fn read_raw_keypoints(&mut self, kp_buf: &wgpu::Buffer, nk: usize) -> Vec<RawKp> {
        if nk == 0 {
            return Vec::new();
        }
        let active_bytes = nk * KP_WORDS * 4;
        if self
            .keypoint_staging
            .as_ref()
            .is_none_or(|staging| staging.size() < active_bytes as u64)
        {
            self.keypoint_staging = Some(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sift-keypoint-staging"),
                size: active_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let staging = self
            .keypoint_staging
            .as_ref()
            .expect("keypoint staging allocated");
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sift-keypoint-readback"),
            });
        encoder.copy_buffer_to_buffer(kp_buf, 0, staging, 0, active_bytes as u64);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..active_bytes as u64);
        map_buffer_async(&self.ctx.device, slice)
            .await
            .expect("map keypoint staging");
        let mapped = slice.get_mapped_range().expect("map oriented keypoints");
        let oriented = &mapped[..active_bytes];
        let raw = oriented
            .chunks_exact(KP_WORDS * 4)
            .map(|record| {
                let word = |index: usize| {
                    let start = index * 4;
                    u32::from_ne_bytes([
                        record[start],
                        record[start + 1],
                        record[start + 2],
                        record[start + 3],
                    ])
                };
                RawKp {
                    x: word(0),
                    y: word(1),
                    octave: word(2) >> 8,
                    level: word(2) & 0xFF,
                    bin: word(3),
                    orientation: f32::from_bits(word(4)),
                    contrast: f32::from_bits(word(5)),
                    dx: f32::from_bits(word(6)),
                    dy: f32::from_bits(word(7)),
                    ds: f32::from_bits(word(8)),
                }
            })
            .collect();
        drop(mapped);
        staging.unmap();
        raw
    }

    /// Whether `config` selects the path this extractor implements.
    pub fn supports(config: &SiftConfig) -> bool {
        config.detector == SiftDetector::Dog
            && !config.affine
            && !config.multi_anisotropy
            && !config.domain_size_pooling
            && !config.scale_adaptive_gradients
            && !config.vlfeat_compatible_descriptor
            && !config.vlfeat_compatible_detector
            && !config.standard_orientation_peaks
    }

    /// Extract keypoints and descriptors; same contract (ordering, early
    /// octave break, `max_keypoints` truncation) as the CPU `extract_sift`.
    pub fn extract(
        &mut self,
        image: &GrayImage<'_>,
        config: &SiftConfig,
    ) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftGpuError> {
        pollster::block_on(self.extract_async(image, config))
    }

    /// Asynchronously extract keypoints and descriptors.
    ///
    /// Buffer-map completions suspend this future instead of blocking the
    /// calling thread. Detection still requires a host-visible count and
    /// keypoint list for the existing canonical ordering and selection rules.
    pub async fn extract_async(
        &mut self,
        image: &GrayImage<'_>,
        config: &SiftConfig,
    ) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftGpuError> {
        if !Self::supports(config) {
            return Err(SiftGpuError::Unsupported);
        }
        let raw = loop {
            match self.detect(image, config).await? {
                Some(raw) => break raw,
                None => self.cand_div = (self.cand_div / 2).max(1),
            }
        };
        let keypoints = select_keypoints(raw, config);
        let descriptors = self.describe(image, &keypoints, config).await;
        Ok((keypoints, descriptors))
    }

    /// Scale space + extrema, then capped orientation (or full orientation on
    /// fallback). `None` when a capacity overflowed (caller grows and retries).
    async fn detect(
        &mut self,
        image: &GrayImage<'_>,
        config: &SiftConfig,
    ) -> Result<Option<Vec<RawKp>>, SiftGpuError> {
        // Keep owned handles so the staging helpers can borrow `&mut self`
        // while the same device/queue are used for encoding.
        let dev_owned = self.ctx.device.clone();
        let queue_owned = self.ctx.queue.clone();
        let dev = &dev_owned;
        let queue = &queue_owned;
        let (w0, h0) = (image.width, image.height);
        if w0 < 16 || h0 < 16 || image.pixels.len() < w0 * h0 {
            return Err(SiftGpuError::ImageTooSmall {
                width: w0,
                height: h0,
            });
        }
        let intervals = config.intervals.max(1);
        let auto_octaves = ((w0.min(h0) as f64).log2() - 4.0).max(1.0) as usize;
        let num_octaves = if config.octaves == 0 {
            auto_octaves
        } else {
            config.octaves.min(auto_octaves)
        }
        .min(MAX_OCTAVES);
        let levels = intervals + 3;
        let k = 2.0f64.powf(1.0 / intervals as f64);

        // Octave sizes: doubled input, then integer halving.
        let mut dims = Vec::with_capacity(num_octaves);
        let (mut w, mut h) = (w0 * 2, h0 * 2);
        for _ in 0..num_octaves {
            dims.push((w, h));
            w /= 2;
            h /= 2;
        }

        // Blur kernels: [initial, step 1..=intervals+2].
        let mut weights: Vec<f32> = Vec::new();
        let mut kernels: Vec<(u32, u32)> = Vec::new(); // (offset, radius)
        let mut add_kernel = |sigma: f64, weights: &mut Vec<f32>| {
            let kern = gaussian_kernel(sigma);
            kernels.push((weights.len() as u32, (kern.len() / 2) as u32));
            weights.extend_from_slice(&kern);
        };
        add_kernel(
            (config.sigma_base * config.sigma_base - (2.0 * config.sigma_input).powi(2))
                .max(1e-4)
                .sqrt(),
            &mut weights,
        );
        for j in 1..levels {
            let target = config.sigma_base * k.powi(j as i32);
            let prev = config.sigma_base * k.powi(j as i32 - 1);
            add_kernel(
                ((target * target - prev * prev).max(1e-6)).sqrt(),
                &mut weights,
            );
        }

        let storage_use = storage_usage(wgpu::BufferUsages::empty());
        let img_id =
            self.buffers
                .ensure(dev, "sift-image-input", (w0 * h0 * 4) as u64, storage_use);
        let img_buf = self.buffers.handle(img_id);
        queue.write_buffer(&img_buf, 0, bytemuck::cast_slice(&image.pixels[..w0 * h0]));
        let wt_id =
            self.buffers
                .ensure(dev, "sift-weights", (weights.len() * 4) as u64, storage_use);
        let wt_buf = self.buffers.handle(wt_id);
        queue.write_buffer(&wt_buf, 0, bytemuck::cast_slice(&weights));
        let (tw, th) = dims[0];
        let tmp_id = self
            .buffers
            .ensure(dev, "sift-tmp", (tw * th * 4) as u64, storage_use);
        let tmp = self.buffers.handle(tmp_id);
        let gauss_ids: Vec<usize> = dims
            .iter()
            .enumerate()
            .map(|(octave, &(w, h))| {
                self.buffers.ensure(
                    dev,
                    &format!("sift-gauss-{octave}"),
                    (w * h * levels * 4) as u64,
                    storage_use,
                )
            })
            .collect();
        let gauss: Vec<Arc<wgpu::Buffer>> = gauss_ids
            .iter()
            .map(|&id| self.buffers.handle(id))
            .collect();
        let dog_ids: Vec<usize> = dims
            .iter()
            .enumerate()
            .map(|(octave, &(w, h))| {
                self.buffers.ensure(
                    dev,
                    &format!("sift-dog-{octave}"),
                    (w * h * (levels - 1) * 4) as u64,
                    storage_use,
                )
            })
            .collect();
        let dogs: Vec<Arc<wgpu::Buffer>> =
            dog_ids.iter().map(|&id| self.buffers.handle(id)).collect();
        let caps: Vec<usize> = dims
            .iter()
            .map(|&(w, h)| (w * h / self.cand_div).max(4096))
            .collect();
        let cand_ids: Vec<usize> = caps
            .iter()
            .enumerate()
            .map(|(octave, &c)| {
                self.buffers.ensure(
                    dev,
                    &format!("sift-candidates-{octave}"),
                    (c * 28) as u64,
                    storage_use,
                )
            })
            .collect();
        let cands: Vec<Arc<wgpu::Buffer>> =
            cand_ids.iter().map(|&id| self.buffers.handle(id)).collect();
        let kp_cap: usize = caps.iter().sum::<usize>() * 2;
        let kp_id = self.buffers.ensure(
            dev,
            "sift-oriented-keypoints",
            (kp_cap * KP_WORDS * 4) as u64,
            storage_use,
        );
        let kp_buf = self.buffers.handle(kp_id);
        let counters_id = self.buffers.ensure(
            dev,
            "sift-counters",
            (COUNTER_WORDS * 4) as u64,
            storage_use,
        );
        let counters = self.buffers.handle(counters_id);
        queue.write_buffer(&counters, 0, &[0u8; COUNTER_WORDS * 4]);
        let counter_bytes = COUNTER_WORDS * 4;
        if self
            .counter_staging
            .as_ref()
            .is_none_or(|staging| staging.size() < counter_bytes as u64)
        {
            self.counter_staging = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sift-counter-staging"),
                size: counter_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let args_id = self.buffers.ensure(
            dev,
            "sift-args",
            (MAX_OCTAVES * 3 * 4) as u64,
            storage_use | wgpu::BufferUsages::INDIRECT,
        );
        let args = self.buffers.handle(args_id);

        // Record the per-dispatch parameters, then build bind groups.
        let mut params = Params { words: Vec::new() };
        enum Op {
            Image {
                pipe: usize,
                slot: u64,
                octave: usize,
                src: (usize, usize),
                dst: (usize, usize),
                groups: (u32, u32, u32),
            },
            Detect {
                pipe: usize,
                slot: u64,
                octave: usize,
            },
        }
        // Buffer ids: 0 = img, 1 = tmp, 2 + o = gauss[o], 2 + n + o = dogs[o].
        let n = num_octaves;
        let gid = |o: usize| 2 + o;
        let did = |o: usize| 2 + n + o;
        let groups = |w: usize, h: usize, z: usize| {
            ((w as u32).div_ceil(16), (h as u32).div_ceil(16), z as u32)
        };
        let mut ops: Vec<Op> = Vec::new();
        let f = |v: f64| (v as f32).to_bits();
        for (o, &(w, h)) in dims.iter().enumerate() {
            let lvl = w * h;
            let (wu, hu) = (w as u32, h as u32);
            if o == 0 {
                let (wo, r) = kernels[0];
                let slot = params.push(&[wu, hu, 0, lvl as u32, 0, 0, w0 as u32, h0 as u32]);
                let upsample_pipe = if config.aligned_octave0_upsample {
                    5
                } else {
                    0
                };
                ops.push(Op::Image {
                    octave: o,
                    pipe: upsample_pipe,
                    slot,
                    src: (0, 0),
                    dst: (gid(0), 0),
                    groups: groups(w, h, 1),
                });
                let slot = params.push(&[wu, hu, lvl as u32, 0, r, wo, wu, hu]);
                ops.push(Op::Image {
                    octave: o,
                    pipe: 1,
                    slot,
                    src: (gid(0), 0),
                    dst: (1, 0),
                    groups: groups(w, h, 1),
                });
                let slot = params.push(&[wu, hu, 0, 0, r, wo, wu, hu]);
                ops.push(Op::Image {
                    octave: o,
                    pipe: 2,
                    slot,
                    src: (1, 0),
                    dst: (gid(0), 0),
                    groups: groups(w, h, 1),
                });
            } else {
                let (pw, ph) = dims[o - 1];
                let slot = params.push(&[
                    wu,
                    hu,
                    (intervals * pw * ph) as u32,
                    0,
                    0,
                    0,
                    pw as u32,
                    ph as u32,
                ]);
                ops.push(Op::Image {
                    octave: o,
                    pipe: 3,
                    slot,
                    src: (gid(o - 1), 0),
                    dst: (gid(o), 0),
                    groups: groups(w, h, 1),
                });
            }
            for (j, &(wo, r)) in kernels.iter().enumerate().skip(1) {
                let slot = params.push(&[wu, hu, ((j - 1) * lvl) as u32, 0, r, wo, wu, hu]);
                ops.push(Op::Image {
                    octave: o,
                    pipe: 1,
                    slot,
                    src: (gid(o), 0),
                    dst: (1, 0),
                    groups: groups(w, h, 1),
                });
                let slot = params.push(&[
                    wu,
                    hu,
                    0,
                    (j * lvl) as u32,
                    r,
                    wo,
                    wu,
                    hu,
                    ((j - 1) * lvl) as u32,
                    ((j - 1) * lvl) as u32,
                ]);
                ops.push(Op::Image {
                    octave: o,
                    pipe: 6,
                    slot,
                    src: (1, gid(o)),
                    dst: (gid(o), did(o)),
                    groups: groups(w, h, 1),
                });
            }
            let slot = params.push(&[
                wu,
                hu,
                o as u32,
                caps[o] as u32,
                f(config.contrast_threshold),
                f(config.edge_threshold),
                f(1.5 * config.sigma_base),
                f(k),
                kp_cap as u32,
                config.max_orientations as u32,
                config.subpixel_localization as u32,
                (levels - 1) as u32,
            ]);
            ops.push(Op::Detect {
                pipe: 0,
                slot,
                octave: o,
            });
            ops.push(Op::Detect {
                pipe: 1,
                slot,
                octave: o,
            });
            ops.push(Op::Detect {
                pipe: 2,
                slot,
                octave: o,
            });
        }
        let param_id = self.buffers.ensure(
            dev,
            "sift-detect-params",
            (params.words.len() * 4) as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let param_buf = self.buffers.handle(param_id);
        queue.write_buffer(&param_buf, 0, bytemuck::cast_slice(&params.words));
        let uniform = |slot: u64| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &param_buf,
                offset: slot,
                size: NonZeroU64::new(SLOT_BIND),
            })
        };
        let buf = |id: usize| -> &wgpu::Buffer {
            match id {
                0 => &img_buf,
                1 => &tmp,
                id if id < 2 + n => &gauss[id - 2],
                id => &dogs[id - 2 - n],
            }
        };
        // Owned handles keep `self` free for the staging helpers below.
        let image_pipes = [
            self.pipes.upsample2x.clone(),
            self.pipes.blur_h.clone(),
            self.pipes.blur_v.clone(),
            self.pipes.halve.clone(),
            self.pipes.dog.clone(),
            self.pipes.upsample2x_aligned.clone(),
            self.pipes.blur_v_dog.clone(),
        ];
        let detect_pipes = [
            self.pipes.extrema.clone(),
            self.pipes.orientation_args.clone(),
            self.pipes.orientation.clone(),
        ];

        macro_rules! encode_ops {
            ($pass:ident, $phase:expr, $min_octave:expr, $max_octave:expr) => {
                #[allow(unused_comparisons)]
                for op in &ops {
                    match *op {
                        Op::Image {
                            pipe,
                            slot,
                            octave,
                            src,
                            dst,
                            groups,
                        } if ($phase == 0 || $phase == 3)
                            && octave >= $min_octave
                            && octave < $max_octave =>
                        {
                            let pl = &image_pipes[pipe];
                            let mut entries = vec![
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: uniform(slot),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: buf(src.0).as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: buf(dst.0).as_entire_binding(),
                                },
                            ];
                            if pipe == 1 || pipe == 2 || pipe == 6 {
                                entries.push(wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wt_buf.as_entire_binding(),
                                });
                            }
                            if pipe == 6 {
                                entries.push(wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: buf(dst.1).as_entire_binding(),
                                });
                            }
                            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("sift-image"),
                                layout: &pl.get_bind_group_layout(0),
                                entries: &entries,
                            });
                            $pass.set_pipeline(pl);
                            $pass.set_bind_group(0, &bg, &[]);
                            // blur_h covers 64 columns and blur_v /
                            // blur_v_dog cover 64 rows per 16x16 workgroup.
                            let (gx, gy) = if pipe == 1 {
                                (groups.0.div_ceil(4), groups.1)
                            } else if pipe == 2 || pipe == 6 {
                                (groups.0, groups.1.div_ceil(4))
                            } else {
                                (groups.0, groups.1)
                            };
                            $pass.dispatch_workgroups(gx, gy, groups.2);
                        }
                        Op::Detect { pipe, slot, octave }
                            if (($phase == 1 && pipe == 0)
                                || ($phase == 2 && pipe != 0)
                                || $phase == 3)
                                && octave >= $min_octave
                                && octave < $max_octave =>
                        {
                            let pl = &detect_pipes[pipe];
                            let mut entries = vec![wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform(slot),
                            }];
                            let bufs: Vec<(u32, &wgpu::Buffer)> = match pipe {
                                0 => vec![(1, &dogs[octave]), (2, &cands[octave]), (3, &counters)],
                                1 => vec![(3, &counters), (5, &args)],
                                _ => vec![
                                    (1, &dogs[octave]),
                                    (2, &cands[octave]),
                                    (3, &counters),
                                    (4, &kp_buf),
                                ],
                            };
                            for (binding, b) in bufs {
                                entries.push(wgpu::BindGroupEntry {
                                    binding,
                                    resource: b.as_entire_binding(),
                                });
                            }
                            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("sift-detect"),
                                layout: &pl.get_bind_group_layout(0),
                                entries: &entries,
                            });
                            $pass.set_pipeline(pl);
                            $pass.set_bind_group(0, &bg, &[]);
                            match pipe {
                                0 => {
                                    let (w, h) = dims[octave];
                                    let g = groups(w, h, 1);
                                    $pass.dispatch_workgroups(g.0, g.1, 1);
                                }
                                1 => $pass.dispatch_workgroups(1, 1, 1),
                                _ => {
                                    $pass.dispatch_workgroups_indirect(&args, (octave * 12) as u64)
                                }
                            }
                        }
                        _ => {}
                    }
                }
            };
        }

        let max = config.max_keypoints;
        let early_break = !config.prefer_larger_scale && !config.full_pyramid;
        // On the default capped path the CPU stops as soon as octave 0 alone
        // fills the budget. Build later octaves only when octave 0 cannot.
        let octave0_only = early_break && max != usize::MAX && n > 1;
        let phase_a_end = if octave0_only { 1 } else { n };

        // Phase A: scale space + extrema. Orientation is submitted separately
        // so capped runs can preselect candidates first.
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sift-detect-phase-a"),
        });
        if self.timestamps.is_some() {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sift-scale-space"),
                    timestamp_writes: self.timestamps.as_ref().map(|p| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &p.queries,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }
                    }),
                });
                encode_ops!(pass, 0, 0, phase_a_end);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sift-extrema"),
                    timestamp_writes: self.timestamps.as_ref().map(|p| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &p.queries,
                            beginning_of_pass_write_index: Some(2),
                            end_of_pass_write_index: Some(3),
                        }
                    }),
                });
                encode_ops!(pass, 1, 0, phase_a_end);
            }
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sift-detect-phase-a"),
                timestamp_writes: None,
            });
            encode_ops!(pass, 0, 0, phase_a_end);
            encode_ops!(pass, 1, 0, phase_a_end);
        }
        {
            let counter_staging = self
                .counter_staging
                .as_ref()
                .expect("counter staging allocated");
            encoder.copy_buffer_to_buffer(&counters, 0, counter_staging, 0, counter_bytes as u64);
        }
        queue.submit(Some(encoder.finish()));

        let mut counts = self.map_counter_staging().await;
        if std::env::var_os("SIFT_GPU_DEBUG_COUNTS").is_some() {
            eprintln!(
                "sift counts candidates={:?} caps={:?}",
                &counts[..n],
                &caps[..n]
            );
        }
        if (0..phase_a_end).any(|o| counts[o] as usize > caps[o]) {
            return Ok(None);
        }

        if max == 0 {
            return Ok(Some(Vec::new()));
        }
        let total_candidates: usize = counts[..phase_a_end].iter().map(|&c| c as usize).sum();
        if total_candidates == 0 && !octave0_only {
            return Ok(Some(Vec::new()));
        }

        // On the default early-break path the CPU stops as soon as octave 0
        // alone fills the budget, so capping within octave 0 is sufficient.
        // Full-pyramid / larger-scale modes walk every octave and rank
        // globally instead.
        if max != usize::MAX && (total_candidates > max || octave0_only) {
            let pool_octaves = if octave0_only { 1 } else { n };
            let candidates = self.read_candidates(&cands, &counts, pool_octaves).await;
            let pool = if early_break {
                candidates
                    .into_iter()
                    .filter(|candidate| candidate.octave == 0)
                    .collect()
            } else {
                candidates
            };
            let selected = select_candidates(pool, config);
            if !selected.is_empty() {
                let mut groups: [Vec<GpuCandidate>; MAX_OCTAVES] =
                    std::array::from_fn(|_| Vec::new());
                for candidate in selected {
                    let octave = candidate.octave as usize;
                    if let Some(group) = groups.get_mut(octave) {
                        group.push(GpuCandidate {
                            x: candidate.x,
                            y: candidate.y,
                            level: candidate.level,
                            value: candidate.value,
                            dx: candidate.dx,
                            dy: candidate.dy,
                            ds: candidate.ds,
                        });
                    }
                }
                let mut selected_counts = [0usize; MAX_OCTAVES];
                let mut compact_ids: [Option<usize>; MAX_OCTAVES] = [None; MAX_OCTAVES];
                for (octave, group) in groups.iter().enumerate().take(n) {
                    if group.is_empty() {
                        continue;
                    }
                    let bytes = group.len() * CAND_BYTES;
                    let id = self.buffers.ensure(
                        dev,
                        &format!("sift-selected-candidates-{octave}"),
                        bytes as u64,
                        storage_use,
                    );
                    let buffer = self.buffers.handle(id);
                    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(group));
                    compact_ids[octave] = Some(id);
                    selected_counts[octave] = group.len();
                }

                if std::env::var_os("SIFT_GPU_DEBUG_COUNTS").is_some() {
                    eprintln!(
                        "sift precap early_break={early_break} selected={:?}",
                        &selected_counts[..n]
                    );
                }
                let mut compact_words = vec![0u32; MAX_OCTAVES * SLOT_WORDS];
                for octave in 0..n {
                    if selected_counts[octave] == 0 {
                        continue;
                    }
                    let (ow, oh) = dims[octave];
                    let base = octave * SLOT_WORDS;
                    compact_words[base..base + 12].copy_from_slice(&[
                        ow as u32,
                        oh as u32,
                        octave as u32,
                        selected_counts[octave] as u32,
                        f(config.contrast_threshold),
                        f(config.edge_threshold),
                        f(1.5 * config.sigma_base),
                        f(k),
                        kp_cap as u32,
                        config.max_orientations as u32,
                        config.subpixel_localization as u32,
                        (levels - 1) as u32,
                    ]);
                }
                let compact_param_id = self.buffers.ensure(
                    dev,
                    "sift-compact-params",
                    (MAX_OCTAVES as u64) * SLOT,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                );
                let compact_param_buf = self.buffers.handle(compact_param_id);
                queue.write_buffer(&compact_param_buf, 0, bytemuck::cast_slice(&compact_words));

                queue.write_buffer(&counters, (KP_COUNTER * 4) as u64, &[0u8; 4]);
                let mut orient = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sift-orientation-capped"),
                });
                {
                    let mut pass = orient.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("sift-orientation-capped"),
                        timestamp_writes: self.timestamps.as_ref().map(|p| {
                            wgpu::ComputePassTimestampWrites {
                                query_set: &p.queries,
                                beginning_of_pass_write_index: Some(4),
                                end_of_pass_write_index: Some(5),
                            }
                        }),
                    });
                    for octave in 0..n {
                        let count = selected_counts[octave];
                        if count == 0 {
                            continue;
                        }
                        let compact_id = compact_ids[octave].expect("selected candidates uploaded");
                        let compact_buf = self.buffers.handle(compact_id);
                        let entries = [
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &compact_param_buf,
                                    offset: (octave as u64) * SLOT,
                                    size: NonZeroU64::new(SLOT_BIND),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: dogs[octave].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: compact_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: counters.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: kp_buf.as_entire_binding(),
                            },
                        ];
                        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("sift-orientation-capped"),
                            layout: &self.pipes.orientation.get_bind_group_layout(0),
                            entries: &entries,
                        });
                        pass.set_pipeline(&self.pipes.orientation);
                        pass.set_bind_group(0, &bind_group, &[]);
                        let (gx, gy) = dispatch_2d(count as u32);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
                {
                    let counter_staging = self
                        .counter_staging
                        .as_ref()
                        .expect("counter staging allocated");
                    orient.copy_buffer_to_buffer(
                        &counters,
                        0,
                        counter_staging,
                        0,
                        counter_bytes as u64,
                    );
                }
                queue.submit(Some(orient.finish()));

                let fast_counts = self.map_counter_staging().await;
                let nk_fast = fast_counts[KP_COUNTER] as usize;
                if std::env::var_os("SIFT_GPU_DEBUG_COUNTS").is_some() {
                    eprintln!("sift fast oriented={nk_fast} max={max}");
                }
                if nk_fast > kp_cap {
                    return Ok(None);
                }
                if nk_fast >= max {
                    let raw = self.read_raw_keypoints(&kp_buf, nk_fast).await;
                    return Ok(Some(raw));
                }
            }
        }

        if octave0_only {
            // Octave 0 did not fill the budget. Build the remaining octaves
            // only now; they are not needed on the common fast path.
            let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sift-detect-phase-a-rest"),
            });
            if self.timestamps.is_some() {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("sift-scale-space-rest"),
                        timestamp_writes: self.timestamps.as_ref().map(|p| {
                            wgpu::ComputePassTimestampWrites {
                                query_set: &p.queries,
                                beginning_of_pass_write_index: Some(0),
                                end_of_pass_write_index: Some(1),
                            }
                        }),
                    });
                    encode_ops!(pass, 0, 1, n);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("sift-extrema-rest"),
                        timestamp_writes: self.timestamps.as_ref().map(|p| {
                            wgpu::ComputePassTimestampWrites {
                                query_set: &p.queries,
                                beginning_of_pass_write_index: Some(2),
                                end_of_pass_write_index: Some(3),
                            }
                        }),
                    });
                    encode_ops!(pass, 1, 1, n);
                }
            } else {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sift-detect-phase-a-rest"),
                    timestamp_writes: None,
                });
                encode_ops!(pass, 0, 1, n);
                encode_ops!(pass, 1, 1, n);
            }
            {
                let counter_staging = self
                    .counter_staging
                    .as_ref()
                    .expect("counter staging allocated");
                encoder.copy_buffer_to_buffer(
                    &counters,
                    0,
                    counter_staging,
                    0,
                    counter_bytes as u64,
                );
            }
            queue.submit(Some(encoder.finish()));

            counts = self.map_counter_staging().await;
            if std::env::var_os("SIFT_GPU_DEBUG_COUNTS").is_some() {
                eprintln!(
                    "sift counts after octave 0 fallback={:?} caps={:?}",
                    &counts[..n],
                    &caps[..n]
                );
            }
            if (1..n).any(|o| counts[o] as usize > caps[o]) {
                return Ok(None);
            }
        }

        // Fallback / uncapped path: orient every candidate in every octave.
        queue.write_buffer(&counters, (KP_COUNTER * 4) as u64, &[0u8; 4]);
        let mut orient = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sift-orientation"),
        });
        if self.timestamps.is_some() {
            let mut pass = orient.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sift-orientation"),
                timestamp_writes: self.timestamps.as_ref().map(|p| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &p.queries,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }
                }),
            });
            encode_ops!(pass, 2, 0, n);
        } else {
            let mut pass = orient.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sift-orientation"),
                timestamp_writes: None,
            });
            encode_ops!(pass, 2, 0, n);
        }
        {
            let counter_staging = self
                .counter_staging
                .as_ref()
                .expect("counter staging allocated");
            orient.copy_buffer_to_buffer(&counters, 0, counter_staging, 0, counter_bytes as u64);
        }
        queue.submit(Some(orient.finish()));

        let counts = self.map_counter_staging().await;
        if (0..n).any(|o| counts[o] as usize > caps[o]) || counts[KP_COUNTER] as usize > kp_cap {
            return Ok(None);
        }
        let nk = counts[KP_COUNTER] as usize;
        let raw = self.read_raw_keypoints(&kp_buf, nk).await;
        Ok(Some(raw))
    }

    async fn describe(
        &mut self,
        image: &GrayImage<'_>,
        keypoints: &[SiftKeypoint],
        config: &SiftConfig,
    ) -> Vec<Vec<f32>> {
        let nk = keypoints.len();
        if nk == 0 {
            return Vec::new();
        }
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let (w0, h0) = (image.width, image.height);
        let use_ = wgpu::BufferUsages::empty();
        let image_id = self.buffers.ensure(
            dev,
            "sift-image-input",
            (w0 * h0 * 4) as u64,
            storage_usage(use_),
        );
        let img_buf = self.buffers.handle(image_id);
        let kp_data: Vec<f32> = keypoints
            .iter()
            .flat_map(|k| [k.x as f32, k.y as f32, k.sigma as f32, k.orientation as f32])
            .collect();
        let kp_id = self.buffers.ensure(
            dev,
            "sift-desc-kps",
            (kp_data.len() * 4) as u64,
            storage_usage(use_),
        );
        let kp_buf = self.buffers.handle(kp_id);
        queue.write_buffer(&kp_buf, 0, bytemuck::cast_slice(&kp_data));
        let out_id =
            self.buffers
                .ensure(dev, "sift-desc", (nk * 128 * 4) as u64, storage_usage(use_));
        let out = self.buffers.handle(out_id);
        let norm = match config.normalization {
            SiftNormalization::L2 => 0u32,
            SiftNormalization::L1Root => 1u32,
        };
        let pw: [u32; 8] = [
            nk as u32,
            w0 as u32,
            h0 as u32,
            norm,
            (config.descriptor_magnification as f32).to_bits(),
            0,
            0,
            0,
        ];
        let param_id = self.buffers.ensure(
            dev,
            "sift-desc-params",
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let param_buf = self.buffers.handle(param_id);
        queue.write_buffer(&param_buf, 0, bytemuck::cast_slice(&pw));
        let pl = &self.pipes.describe;
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sift-describe"),
            layout: &pl.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: img_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sift-describe"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sift-describe"),
                timestamp_writes: self.timestamps.as_ref().map(|p| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &p.queries,
                        beginning_of_pass_write_index: Some(6),
                        end_of_pass_write_index: Some(7),
                    }
                }),
            });
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &bg, &[]);
            let (gx, gy) = dispatch_2d(nk as u32);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit(Some(encoder.finish()));
        let bytes = nk * 128 * 4;
        let staging_id = self.buffers.ensure(
            dev,
            "sift-descriptor-staging",
            bytes as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging = self.buffers.handle(staging_id);
        let mut readback = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sift-descriptor-readback"),
        });
        readback.copy_buffer_to_buffer(&out, 0, &staging, 0, bytes as u64);
        queue.submit(Some(readback.finish()));
        let slice = staging.slice(..bytes as u64);
        let _ = map_buffer_async(dev, slice).await;
        let mapped = slice.get_mapped_range().expect("map descriptor output");
        let floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        floats.chunks_exact(128).map(|c| c.to_vec()).collect()
    }
}

/// Map raw detections to original-frame keypoints in the CPU's order, then
/// apply its early octave break and `max_keypoints` truncation.
///
/// The common capped path never needs a full canonical sort: the octave cutoff
/// only needs per-octave counts, and the retained set only needs its strongest
/// `max_keypoints` entries. A linear-time selection plus a sort of the small
/// selected prefix replaces two sorts over every oriented keypoint.
fn select_keypoints(raw: Vec<RawKp>, config: &SiftConfig) -> Vec<SiftKeypoint> {
    let max = config.max_keypoints;
    if raw.is_empty() || max == 0 {
        return Vec::new();
    }
    let intervals = config.intervals.max(1);
    let k = 2.0f64.powf(1.0 / intervals as f64);
    let map = |r: &RawKp| {
        let upsample = (1usize << r.octave) as f64 / 2.0;
        let sigma = if r.ds != 0.0 && config.subpixel_scale_refine {
            config.sigma_base * k.powf(r.level as f64 + r.ds as f64) * upsample
        } else {
            config.sigma_base * k.powi(r.level as i32) * upsample
        };
        SiftKeypoint::from_location_scale_orientation(
            (r.x as f64 + r.dx as f64) * upsample,
            (r.y as f64 + r.dy as f64) * upsample,
            sigma,
            r.orientation as f64,
        )
    };

    if max == usize::MAX || raw.len() <= max {
        let mut raw = raw;
        raw.sort_by_key(|r| (r.octave, r.level, r.y, r.x, r.bin));
        return raw.iter().map(map).collect();
    }

    // CPU stops after the first octave whose cumulative detections fill the
    // budget; it does not cut inside an octave.
    let mut raw = raw;
    if !config.prefer_larger_scale && !config.full_pyramid {
        let mut counts = [0usize; MAX_OCTAVES];
        for r in &raw {
            if let Some(count) = counts.get_mut(r.octave as usize) {
                *count += 1;
            }
        }
        let mut cumulative = 0usize;
        let mut cutoff = 0u32;
        for (octave, &count) in counts.iter().enumerate() {
            cumulative += count;
            cutoff = octave as u32;
            if cumulative >= max {
                break;
            }
        }
        raw.retain(|r| r.octave <= cutoff);
    }

    let canonical = |r: &RawKp| (r.octave, r.level, r.y, r.x, r.bin);
    let mut raw = raw;
    if !config.prefer_larger_scale {
        // Rank raw records directly so only the selected prefix is mapped.
        let compare = |a: &RawKp, b: &RawKp| {
            b.contrast
                .total_cmp(&a.contrast)
                .then_with(|| canonical(a).cmp(&canonical(b)))
        };
        if raw.len() > max {
            raw.select_nth_unstable_by(max, compare);
            raw.truncate(max);
            raw.sort_by(compare);
        }
        return raw.iter().map(map).collect();
    }

    type Ranked = (SiftKeypoint, f32, (u32, u32, u32, u32, u32));
    let mut ranked: Vec<Ranked> = raw
        .iter()
        .map(|r| (map(r), r.contrast, canonical(r)))
        .collect();
    let compare = |a: &Ranked, b: &Ranked| {
        b.0.sigma
            .total_cmp(&a.0.sigma)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    };
    if ranked.len() > max {
        ranked.select_nth_unstable_by(max, compare);
        ranked.truncate(max);
        ranked.sort_by(compare);
    }
    ranked
        .into_iter()
        .map(|(keypoint, _, _)| keypoint)
        .collect()
}

/// Preselect the candidates that can still contribute to the final capped
/// output. Orientation is independent per candidate, so capping first leaves
/// the host-side final ordering/truncation byte-identical when at least
/// `max_keypoints` orientations survive the cap.
///
/// The caller restricts `candidates` to octave 0 on the default early-break
/// path, where the CPU is guaranteed to stop at octave 0 only if this capped
/// set already fills the budget. For the full-pyramid / larger-scale paths
/// the pool spans all octaves and is ranked globally.
fn select_candidates(candidates: Vec<CandidateRec>, config: &SiftConfig) -> Vec<CandidateRec> {
    let max = config.max_keypoints;
    if candidates.is_empty() || max == 0 {
        return Vec::new();
    }
    if max == usize::MAX || candidates.len() <= max {
        return candidates;
    }

    let intervals = config.intervals.max(1);
    let k = 2.0f64.powf(1.0 / intervals as f64);
    let prefer_scale = config.prefer_larger_scale;

    struct Ranked {
        candidate: CandidateRec,
        sigma: f64,
        contrast: f32,
    }

    let mut ranked: Vec<Ranked> = candidates
        .into_iter()
        .map(|candidate| {
            let upsample = (1usize << candidate.octave) as f64 / 2.0;
            let sigma = if prefer_scale {
                if candidate.ds != 0.0 && config.subpixel_scale_refine {
                    config.sigma_base
                        * k.powf(candidate.level as f64 + candidate.ds as f64)
                        * upsample
                } else {
                    config.sigma_base * k.powi(candidate.level as i32) * upsample
                }
            } else {
                0.0
            };
            Ranked {
                contrast: candidate.value.abs(),
                sigma,
                candidate,
            }
        })
        .collect();

    let canonical = |r: &Ranked| {
        (
            r.candidate.octave,
            r.candidate.level,
            r.candidate.y,
            r.candidate.x,
        )
    };
    let compare = |a: &Ranked, b: &Ranked| {
        if prefer_scale {
            b.sigma
                .total_cmp(&a.sigma)
                .then_with(|| b.contrast.total_cmp(&a.contrast))
                .then_with(|| canonical(a).cmp(&canonical(b)))
        } else {
            b.contrast
                .total_cmp(&a.contrast)
                .then_with(|| canonical(a).cmp(&canonical(b)))
        }
    };

    ranked.select_nth_unstable_by(max, compare);
    // Keep every candidate tied with the first excluded one. A candidate's
    // oriented keypoints all share its canonical (octave, level, y, x)
    // prefix, so this cannot drop a boundary item the final total order keeps.
    if prefer_scale {
        let cutoff = ranked[max].sigma;
        ranked.retain(|r| r.sigma >= cutoff);
    } else {
        let cutoff = ranked[max].contrast;
        ranked.retain(|r| r.contrast >= cutoff);
    }

    ranked.into_iter().map(|r| r.candidate).collect()
}
