//! Basalt's frame-to-frame frontend for one camera on the GPU, with wgpu:
//! WebGPU in the browser, Vulkan or Metal natively. One submission per frame
//! builds the pyramid, tracks the previous frame's points forward and back
//! (`flow.wgsl`), marks the grid cells that still hold a point and detects a
//! FAST corner in each empty one; a single readback brings the tracks and
//! the corners back, and the track bookkeeping here produces the same
//! [`TrackFrameOutput`] as [`visloc_basalt::DirectKltStream`] (same ids,
//! same order, same reject reasons). The arithmetic is plain f32, so points
//! agree with the CPU frontend to rounding, not bit for bit.
//!
//! A frame is [`GpuFlow::submit`]ted with a callback for when its results
//! can be read, then [`GpuFlow::finish`]ed: in the browser the callback
//! resolves a promise the worker awaits; natively [`GpuFlow::wait`] blocks.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use visloc_basalt::{
    AffineCompact2f, BasaltFrame, DirectKltConfig, FrameId, KltFailure, RejectReason,
    RejectReasonCounters, TrackFrameOutput, TrackId, TrackObservation,
};

const SHADER: &str = include_str!("flow.wgsl");
/// Largest grid cell `fast` handles (its workgroup score array).
const MAX_CELL: usize = 60;
const MAX_LEVELS: usize = 8;
const PARAMS_BYTES: u64 = 192;
const TRACK_BYTES: u64 = 32;
const CORNER_BYTES: u64 = 8;

/// A GPU device and queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The adapter's name and backend, for logs.
    pub description: String,
    /// The first error the device reported outside a call (an invalid
    /// pipeline, say: the browser's shader compiler is stricter than
    /// wgpu's own), which voids every later submission.
    error: Arc<Mutex<Option<String>>>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

fn report(message: &str) {
    #[cfg(target_arch = "wasm32")]
    console_error(message);
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
}

impl GpuContext {
    /// The default adapter (WebGPU in the browser).
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("visloc-wasm"),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU device: {e}"))?;
        let error = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&error);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let message = format!("GPU frontend: {e}");
            report(&message);
            let mut first = sink.lock().unwrap_or_else(|p| p.into_inner());
            first.get_or_insert(message);
        }));
        Ok(Self {
            device,
            queue,
            description: format!("{} ({:?})", info.name, info.backend),
            error,
        })
    }

    /// The first error the device reported, if any.
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// [`Self::new`], blocking (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking() -> Result<Self, String> {
        pollster::block_on(Self::new())
    }
}

#[derive(Debug, Clone, Copy)]
struct Track {
    linear: [f32; 4],
    position: [f32; 2],
    created: FrameId,
}

/// The frame between [`GpuFlow::submit`] and [`GpuFlow::finish`].
struct Pending {
    frame: BasaltFrame,
    new_half: u32,
    /// The tracks sent, in buffer order.
    sent: Vec<TrackId>,
    bytes: u64,
}

struct Pipelines {
    unpack: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    klt: wgpu::ComputePipeline,
    occupancy: wgpu::ComputePipeline,
    fast: wgpu::ComputePipeline,
}

struct Buffers {
    params: wgpu::Buffer,
    pyramid: wgpu::Buffer,
    image: wgpu::Buffer,
    tracks: wgpu::Buffer,
    results: wgpu::Buffer,
    occupied: wgpu::Buffer,
    corners: wgpu::Buffer,
    staging: wgpu::Buffer,
    down_levels: Vec<wgpu::Buffer>,
}

struct BindGroups {
    unpack: wgpu::BindGroup,
    down: wgpu::BindGroup,
    down_levels: Vec<wgpu::BindGroup>,
    klt: wgpu::BindGroup,
    occupancy: wgpu::BindGroup,
    fast: wgpu::BindGroup,
}

/// The frontend on the GPU (see the module docs).
pub struct GpuFlow {
    ctx: GpuContext,
    config: DirectKltConfig,
    width: usize,
    height: usize,
    /// Per level: width, height, offset within a pyramid.
    levels: Vec<[u32; 3]>,
    pyramid_len: u32,
    cells_x: u32,
    cells_y: u32,
    x_start: u32,
    y_start: u32,
    pipelines: Pipelines,
    buffers: Buffers,
    groups: BindGroups,
    track_capacity: u64,
    /// Which half of the pyramid buffer holds the newest frame.
    current: u32,
    has_previous: bool,
    tracks: BTreeMap<TrackId, Track>,
    next_track_id: TrackId,
    previous_frame: Option<BasaltFrame>,
    pending: Option<Pending>,
}

impl GpuFlow {
    /// A frontend for `width x height` frames with `config`'s pyramid, KLT
    /// and FAST settings (one camera; no gyro seed).
    pub fn new(
        ctx: GpuContext,
        width: usize,
        height: usize,
        config: DirectKltConfig,
    ) -> Result<Self, String> {
        let cell = config.fast.cell_size;
        if cell == 0 || cell > MAX_CELL || config.fast.points_per_cell != 1 {
            return Err(format!(
                "the GPU frontend needs 1 point per cell and cells of at most {MAX_CELL} px"
            ));
        }
        if config.pyramid_levels + 1 > MAX_LEVELS {
            return Err(format!("the GPU frontend has at most {MAX_LEVELS} levels"));
        }
        let mut levels = Vec::new();
        let (mut w, mut h, mut offset) = (width, height, 0usize);
        for level in 0..=config.pyramid_levels {
            if level > 0 {
                if w < 3 || h < 3 {
                    return Err("image too small for the pyramid".into());
                }
                w /= 2;
                h /= 2;
            }
            levels.push([w as u32, h as u32, offset as u32]);
            offset += w * h;
        }
        let pyramid_len = offset as u32;
        let cells_x = (width / cell) as u32;
        let cells_y = (height / cell) as u32;
        let x_start = ((width % cell) / 2) as u32;
        let y_start = ((height % cell) / 2) as u32;

        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flow"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            unpack: pipeline("unpack"),
            down: pipeline("down"),
            klt: pipeline("klt"),
            occupancy: pipeline("occupancy"),
            fast: pipeline("fast"),
        };
        let storage = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16).div_ceil(16) * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cells = u64::from(cells_x * cells_y);
        let track_capacity = 1024;
        let buffers = Buffers {
            params: uniform("flow-params", PARAMS_BYTES),
            pyramid: storage("flow-pyramid", 2 * 4 * u64::from(pyramid_len)),
            image: storage("flow-image", (width * height).div_ceil(2) as u64 * 4),
            tracks: storage("flow-tracks", track_capacity * TRACK_BYTES),
            results: storage("flow-results", track_capacity * TRACK_BYTES),
            occupied: storage("flow-occupied", cells * 4),
            corners: storage("flow-corners", cells * CORNER_BYTES),
            staging: staging(device, track_capacity, cells),
            down_levels: (0..levels.len())
                .map(|level| {
                    let buffer = uniform("flow-down-level", 16);
                    ctx.queue
                        .write_buffer(&buffer, 0, &words(&[level as u32, 0, 0, 0]));
                    buffer
                })
                .collect(),
        };
        let groups = bind_groups(device, &pipelines, &buffers);
        Ok(Self {
            ctx,
            config,
            width,
            height,
            levels,
            pyramid_len,
            cells_x,
            cells_y,
            x_start,
            y_start,
            pipelines,
            buffers,
            groups,
            track_capacity,
            current: 0,
            has_previous: false,
            tracks: BTreeMap::new(),
            next_track_id: 0,
            previous_frame: None,
            pending: None,
        })
    }

    pub fn description(&self) -> &str {
        &self.ctx.description
    }

    /// Forgets the tracks and the previous frame (track ids keep counting).
    pub fn reset(&mut self) {
        if self.pending.take().is_some() {
            self.buffers.staging.unmap();
        }
        self.tracks.clear();
        self.has_previous = false;
        self.previous_frame = None;
    }

    /// Starts tracking a frame of the CPU pyramid's u16 pixels; `on_mapped`
    /// runs once its results can be read ([`Self::finish`]).
    pub fn submit<F>(
        &mut self,
        frame_id: FrameId,
        timestamp_ns: i64,
        pixels: &[u16],
        on_mapped: F,
    ) -> Result<(), String>
    where
        F: FnOnce(Result<(), wgpu::BufferAsyncError>) + wgpu::WasmNotSend + 'static,
    {
        if self.pending.is_some() {
            return Err("the previous frame is not finished".into());
        }
        if pixels.len() != self.width * self.height {
            return Err("frame size does not match the GPU frontend".into());
        }
        if let Some(previous) = self.previous_frame {
            if frame_id <= previous.frame_id || timestamp_ns <= previous.timestamp_ns {
                return Err("frame id/timestamp is not newer than the previous frame".into());
            }
        }
        let new_half = if self.has_previous {
            1 - self.current
        } else {
            self.current
        };
        let sent: Vec<TrackId> = if self.has_previous {
            self.tracks.keys().copied().collect()
        } else {
            Vec::new()
        };
        let track_count = sent.len() as u64;
        if track_count > self.track_capacity {
            self.grow(track_count);
        }
        let queue = &self.ctx.queue;
        if !sent.is_empty() {
            let mut data = Vec::with_capacity(sent.len() * TRACK_BYTES as usize);
            for id in &sent {
                let track = &self.tracks[id];
                for v in track
                    .linear
                    .iter()
                    .chain(&track.position)
                    .chain(&[0.0, 0.0])
                {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
            queue.write_buffer(&self.buffers.tracks, 0, &data);
        }
        queue.write_buffer(
            &self.buffers.params,
            0,
            &self.params(new_half, track_count as u32),
        );
        let mut image = Vec::with_capacity(pixels.len().div_ceil(2) * 4);
        for pair in pixels.chunks(2) {
            let word = u32::from(pair[0]) | (u32::from(*pair.get(1).unwrap_or(&0)) << 16);
            image.extend_from_slice(&word.to_le_bytes());
        }
        queue.write_buffer(&self.buffers.image, 0, &image);

        let device = &self.ctx.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flow"),
        });
        encoder.clear_buffer(&self.buffers.occupied, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flow"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.unpack);
            pass.set_bind_group(0, &self.groups.unpack, &[]);
            pass.dispatch_workgroups(((self.width * self.height) as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(&self.pipelines.down);
            pass.set_bind_group(0, &self.groups.down, &[]);
            for (level, dims) in self.levels.iter().enumerate().skip(1) {
                pass.set_bind_group(1, &self.groups.down_levels[level], &[]);
                pass.dispatch_workgroups(dims[0].div_ceil(16), dims[1].div_ceil(16), 1);
            }
            if track_count > 0 {
                let n = track_count as u32;
                pass.set_pipeline(&self.pipelines.klt);
                pass.set_bind_group(0, &self.groups.klt, &[]);
                pass.dispatch_workgroups(n.min(65535), n.div_ceil(65535), 1);
                pass.set_pipeline(&self.pipelines.occupancy);
                pass.set_bind_group(0, &self.groups.occupancy, &[]);
                pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
            }
            pass.set_pipeline(&self.pipelines.fast);
            pass.set_bind_group(0, &self.groups.fast, &[]);
            pass.dispatch_workgroups(self.cells_x * self.cells_y, 1, 1);
        }
        let results_bytes = track_count * TRACK_BYTES;
        let corner_bytes = u64::from(self.cells_x * self.cells_y) * CORNER_BYTES;
        if results_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.buffers.results,
                0,
                &self.buffers.staging,
                0,
                results_bytes,
            );
        }
        let corner_offset = self.track_capacity * TRACK_BYTES;
        encoder.copy_buffer_to_buffer(
            &self.buffers.corners,
            0,
            &self.buffers.staging,
            corner_offset,
            corner_bytes,
        );
        queue.submit(Some(encoder.finish()));
        let bytes = corner_offset + corner_bytes;
        self.buffers
            .staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, on_mapped);
        self.pending = Some(Pending {
            frame: BasaltFrame::new(frame_id, timestamp_ns, 0),
            new_half,
            sent,
            bytes,
        });
        Ok(())
    }

    /// Blocks until the submitted frame's results can be read (native).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// The submitted frame's tracks, once `on_mapped` has run.
    pub fn finish(&mut self) -> Result<TrackFrameOutput, String> {
        let pending = self.pending.take().ok_or("no frame was submitted")?;
        if let Some(error) = self.ctx.error() {
            self.buffers.staging.unmap();
            return Err(error);
        }
        let data = {
            let view = self
                .buffers
                .staging
                .slice(..pending.bytes)
                .get_mapped_range()
                .map_err(|e| format!("GPU results: {e}"))?;
            view.to_vec()
        };
        self.buffers.staging.unmap();
        let word = |offset: usize| {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };
        let float = |offset: usize| f32::from_bits(word(offset));

        let mut counters = RejectReasonCounters::default();
        let mut retained = Vec::new();
        let mut rejected = Vec::new();
        for (k, id) in pending.sent.iter().enumerate() {
            let base = k * TRACK_BYTES as usize;
            let status = word(base + 24);
            if status == 0 {
                let track = self.tracks.get_mut(id).expect("sent tracks exist");
                track.linear = [
                    float(base),
                    float(base + 4),
                    float(base + 8),
                    float(base + 12),
                ];
                track.position = [float(base + 16), float(base + 20)];
                retained.push(*id);
            } else {
                self.tracks.remove(id);
                counters.record(reject_reason(status));
                rejected.push(*id);
            }
        }
        let frame = pending.frame;
        let corner_base = (self.track_capacity * TRACK_BYTES) as usize;
        let mut created = Vec::new();
        for cell in 0..(self.cells_x * self.cells_y) as usize {
            let offset = corner_base + cell * CORNER_BYTES as usize;
            let (x, y) = (word(offset) as i32, word(offset + 4) as i32);
            if x < 0 || y < 0 {
                continue;
            }
            let id = self.next_track_id;
            self.next_track_id += 1;
            self.tracks.insert(
                id,
                Track {
                    linear: [1.0, 0.0, 0.0, 1.0],
                    position: [x as f32, y as f32],
                    created: frame.frame_id,
                },
            );
            created.push(id);
        }
        if created.is_empty() {
            counters.record(RejectReason::FastNoCandidate);
        }
        let observations = self
            .tracks
            .iter()
            .map(|(&track_id, track)| TrackObservation {
                track_id,
                frame_id: frame.frame_id,
                timestamp_ns: frame.timestamp_ns,
                camera_id: 0,
                pixel: nalgebra::Point2::new(
                    f64::from(track.position[0]),
                    f64::from(track.position[1]),
                ),
            })
            .collect();
        self.current = pending.new_half;
        self.has_previous = true;
        self.previous_frame = Some(frame);
        Ok(TrackFrameOutput {
            frame,
            observations,
            created_track_ids: created,
            retained_track_ids: retained,
            rejected_track_ids: rejected,
            reject_counters: counters,
            stage_trace: Vec::new(),
        })
    }

    /// Replaces the tracks (for comparing with the CPU frontend on the same
    /// input); the previous frame stays the last one submitted.
    pub fn set_tracks(
        &mut self,
        tracks: impl IntoIterator<Item = (TrackId, AffineCompact2f, FrameId)>,
        next_track_id: TrackId,
    ) {
        self.tracks = tracks
            .into_iter()
            .map(|(id, transform, created)| {
                let l = transform.linear();
                let t = transform.translation();
                (
                    id,
                    Track {
                        linear: [l[(0, 0)], l[(0, 1)], l[(1, 0)], l[(1, 1)]],
                        position: [t.x, t.y],
                        created,
                    },
                )
            })
            .collect();
        self.next_track_id = next_track_id;
    }

    /// The tracks' patch transforms, for comparing with the CPU frontend.
    pub fn transforms(&self) -> impl Iterator<Item = (TrackId, AffineCompact2f, FrameId)> + '_ {
        self.tracks.iter().map(|(&id, t)| {
            (
                id,
                AffineCompact2f::new(
                    nalgebra::Matrix2::new(t.linear[0], t.linear[1], t.linear[2], t.linear[3]),
                    nalgebra::Vector2::new(t.position[0], t.position[1]),
                ),
                t.created,
            )
        })
    }

    fn params(&self, new_half: u32, track_count: u32) -> Vec<u8> {
        let mut levels = [0u32; 4 * MAX_LEVELS];
        for (i, level) in self.levels.iter().enumerate() {
            levels[4 * i..4 * i + 3].copy_from_slice(level);
        }
        let fast = &self.config.fast;
        let mut out = words(&levels);
        out.extend(words(&[
            self.levels.len() as u32,
            (1 - new_half) * self.pyramid_len,
            new_half * self.pyramid_len,
            track_count,
            self.config.max_iterations as u32,
            self.config.fb_squared_threshold.to_bits(),
            fast.cell_size as u32,
            self.cells_x,
            self.cells_y,
            self.x_start,
            self.y_start,
            u32::from(fast.threshold),
            u32::from(fast.min_threshold),
            (fast.edge_threshold as f32).to_bits(),
            0,
            0,
        ]));
        debug_assert_eq!(out.len() as u64, PARAMS_BYTES);
        out
    }

    fn grow(&mut self, tracks: u64) {
        let device = &self.ctx.device;
        self.track_capacity = tracks.next_power_of_two();
        let storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: self.track_capacity * TRACK_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        self.buffers.tracks = storage("flow-tracks");
        self.buffers.results = storage("flow-results");
        self.buffers.staging = staging(
            device,
            self.track_capacity,
            u64::from(self.cells_x * self.cells_y),
        );
        self.groups = bind_groups(device, &self.pipelines, &self.buffers);
    }
}

fn staging(device: &wgpu::Device, tracks: u64, cells: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flow-staging"),
        size: (tracks * TRACK_BYTES + cells * CORNER_BYTES).div_ceil(16) * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind_groups(device: &wgpu::Device, p: &Pipelines, b: &Buffers) -> BindGroups {
    let group = |pipeline: &wgpu::ComputePipeline, index: u32, entries: &[(u32, &wgpu::Buffer)]| {
        let entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(index),
            entries: &entries,
        })
    };
    BindGroups {
        unpack: group(
            &p.unpack,
            0,
            &[(0, &b.params), (1, &b.pyramid), (2, &b.image)],
        ),
        down: group(&p.down, 0, &[(0, &b.params), (1, &b.pyramid)]),
        down_levels: b
            .down_levels
            .iter()
            .map(|level| group(&p.down, 1, &[(0, level)]))
            .collect(),
        klt: group(
            &p.klt,
            0,
            &[
                (0, &b.params),
                (1, &b.pyramid),
                (3, &b.tracks),
                (4, &b.results),
            ],
        ),
        occupancy: group(
            &p.occupancy,
            0,
            &[(0, &b.params), (4, &b.results), (5, &b.occupied)],
        ),
        fast: group(
            &p.fast,
            0,
            &[
                (0, &b.params),
                (1, &b.pyramid),
                (5, &b.occupied),
                (6, &b.corners),
            ],
        ),
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The CPU frontend's reject reason for a `flow.wgsl` status.
fn reject_reason(status: u32) -> RejectReason {
    if status == 32 {
        return RejectReason::FrameFbSquared;
    }
    let failure = match status & 15 {
        1 => KltFailure::SourcePatchInvalid,
        2 => KltFailure::TargetNoValidSamples,
        3 => KltFailure::TargetInsufficientOverlap,
        4 => KltFailure::IncrementNonFinite,
        5 => KltFailure::IncrementTooLarge,
        _ => KltFailure::TargetOutOfBounds,
    };
    if status & 16 != 0 {
        RejectReason::FrameBackward(failure)
    } else {
        RejectReason::FrameForward(failure)
    }
}
