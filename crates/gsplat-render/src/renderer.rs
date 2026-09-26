//! The wgpu forward rasterizer.
//!
//! Pipeline (per rendered frame):
//!
//! 1. `project_forward` — project every gaussian; compact the visible ones and
//!    count the tiles each covers.
//! 2. host readback — `num_visible` / `num_intersections`.
//! 3. host — depth-sort the visible ids, prefix-sum the tile counts.
//! 4. `project_visible` — re-project and evaluate SH colour for each visible
//!    gaussian into `projected_splats`.
//! 5. `map_gaussians` — expand each visible gaussian into `(tile_id, compact)`
//!    entries; then host-sort them by tile id.
//! 6. `tile_offsets` — build per-tile `[start, end)` ranges.
//! 7. `rasterize` — one workgroup per tile composites front-to-back.
//!
//! Stages 3 and 5's sorts run on the host at this stage; a device-side radix
//! sort is a planned follow-up (see `docs/rust_3dgs_plan.md`).

use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::cpu_render::Image;
use visloc_gsplat_core::gaussian::Scene;

use crate::gpu::{GpuContext, GpuError};
use crate::packing::PackedScene;
use crate::shaders;
use crate::uniforms::{tile_bounds, ProjectUniforms, RasterUniforms};

/// A scene uploaded to the GPU once.
///
/// `packed` keeps the scene's metadata (count, SH degree); its host arrays
/// are dropped after upload so a large scene is not held twice.
pub struct GpuScene {
    pub packed: PackedScene,
    transforms: wgpu::Buffer,
    opacity: wgpu::Buffer,
    sh: wgpu::Buffer,
}

fn new_storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl GpuScene {
    pub fn upload(ctx: &GpuContext, scene: &Scene) -> Self {
        Self::upload_packed(ctx, PackedScene::from_scene(scene))
    }

    /// Upload an already packed scene (e.g. a trainer's parameter arrays).
    pub fn upload_packed(ctx: &GpuContext, packed: PackedScene) -> Self {
        let transforms = new_storage(
            &ctx.device,
            "transforms",
            (packed.transforms.len() * 4) as u64,
        );
        ctx.queue
            .write_buffer(&transforms, 0, bytemuck::cast_slice(&packed.transforms));
        let opacity = new_storage(&ctx.device, "opacity", (packed.opacity.len() * 4) as u64);
        ctx.queue
            .write_buffer(&opacity, 0, bytemuck::cast_slice(&packed.opacity));
        let sh = new_storage(&ctx.device, "sh", (packed.sh.len() * 4) as u64);
        ctx.queue
            .write_buffer(&sh, 0, bytemuck::cast_slice(&packed.sh));
        Self {
            packed: PackedScene {
                transforms: Vec::new(),
                opacity: Vec::new(),
                sh: Vec::new(),
                ..packed
            },
            transforms,
            opacity,
            sh,
        }
    }

    /// Wrap parameter buffers that already live on the device (e.g. written by
    /// a trainer's on-device densification) in the forward input layouts.
    pub fn from_buffers(
        transforms: wgpu::Buffer,
        opacity: wgpu::Buffer,
        sh: wgpu::Buffer,
        num_gaussians: usize,
        sh_degree: u32,
    ) -> Self {
        let cpc = (sh_degree + 1) as usize;
        Self {
            packed: PackedScene {
                transforms: Vec::new(),
                opacity: Vec::new(),
                sh: Vec::new(),
                sh_coeffs_per_channel: cpc * cpc,
                num_gaussians,
                sh_degree,
            },
            transforms,
            opacity,
            sh,
        }
    }

    /// Floats in the SH buffer (3 channels x (degree + 1)^2 per gaussian).
    pub fn sh_floats(&self) -> usize {
        self.packed.num_gaussians * 3 * self.packed.sh_coeffs_per_channel
    }

    /// Re-upload the packed arrays (used when the host mutates the scene).
    pub fn reupload(&self, ctx: &GpuContext, packed: &PackedScene) {
        ctx.queue.write_buffer(
            &self.transforms,
            0,
            bytemuck::cast_slice(&packed.transforms),
        );
        ctx.queue
            .write_buffer(&self.opacity, 0, bytemuck::cast_slice(&packed.opacity));
        ctx.queue
            .write_buffer(&self.sh, 0, bytemuck::cast_slice(&packed.sh));
    }
}

/// A named binding used to build layouts and bind groups uniformly.
struct Binding {
    binding: u32,
    ty: wgpu::BufferBindingType,
    buffer: wgpu::Buffer,
}

fn build_layout(device: &wgpu::Device, label: &str, bindings: &[Binding]) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .map(|b| wgpu::BindGroupLayoutEntry {
            binding: b.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: b.ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn build_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bindings: &[Binding],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .map(|b| wgpu::BindGroupEntry {
            binding: b.binding,
            resource: b.buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// A compute pipeline plus the bind group for its single bind group (group 0).
struct Stage {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl Stage {
    /// Point the stage at new buffers (same layout), keeping the pipeline.
    fn rebind(&mut self, device: &wgpu::Device, label: &str, bindings: &[Binding]) {
        self.bind_group = build_bind_group(device, label, &self.layout, bindings);
    }
}

fn build_stage(
    device: &wgpu::Device,
    label: &str,
    kernel_name: &str,
    source: String,
    bindings: &[Binding],
) -> Stage {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = build_layout(device, label, bindings);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(kernel_name),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = build_bind_group(device, label, &layout, bindings);
    Stage {
        pipeline,
        layout,
        bind_group,
    }
}

fn dispatch(pass: &mut wgpu::ComputePass<'_>, stage: &Stage, x: u32) {
    pass.set_pipeline(&stage.pipeline);
    pass.set_bind_group(0, &stage.bind_group, &[]);
    pass.dispatch_workgroups(x.max(1), 1, 1);
}

/// Dispatch `groups` workgroups folded into 2D (x <= 65535); the kernel
/// linearises its workgroup id as `wid.x + wid.y * num_workgroups.x`.
fn dispatch_groups_2d(pass: &mut wgpu::ComputePass<'_>, stage: &Stage, groups: u32) {
    const MAX_DIM: u32 = 65535;
    let groups = groups.max(1);
    let x = groups.min(MAX_DIM);
    pass.set_pipeline(&stage.pipeline);
    pass.set_bind_group(0, &stage.bind_group, &[]);
    pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
}

/// Dispatch one 256-wide invocation per element for `threads` elements, folding
/// the workgroup count into 2D so it can exceed the 65535-per-dimension limit.
/// The kernel must linearise with `gid.x + gid.y * num_workgroups.x * 256`.
fn dispatch_threads(pass: &mut wgpu::ComputePass<'_>, stage: &Stage, threads: u32) {
    const MAX_DIM: u32 = 65535;
    let groups = threads.div_ceil(256).max(1);
    let x = groups.min(MAX_DIM);
    pass.set_pipeline(&stage.pipeline);
    pass.set_bind_group(0, &stage.bind_group, &[]);
    pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
}

/// Per-frame scratch buffers sized from the gaussian count.
struct Scratch {
    global_from_compact: wgpu::Buffer,
    compact_from_global: wgpu::Buffer,
    depths: wgpu::Buffer,
    intersect_counts: wgpu::Buffer,
    cum_tiles_hit: wgpu::Buffer,
    projected_splats: wgpu::Buffer,
    /// Per-isect tile id (the tile sort's keys; `tile_pairs[0].keys`).
    tile_id_from_isect: wgpu::Buffer,
    /// Per-isect compact gaussian id: written by `map` in isect order, sorted
    /// in place by tile, then read by rasterize (`tile_pairs[0].values`).
    compact_sorted: wgpu::Buffer,
    /// `map` also writes each isect's slot; the sort does not need it, so it
    /// lands in the ping-pong buffer the sort overwrites (`tile_pairs[1].values`).
    isect_id: wgpu::Buffer,
    num_visible: wgpu::Buffer,
    num_intersections: wgpu::Buffer,
    readback: wgpu::Buffer,
    max_isects: usize,
}

impl Scratch {
    fn new(
        device: &wgpu::Device,
        n: usize,
        max_isects: usize,
        tile_pairs: &[crate::sort::SortBuffers; 2],
    ) -> Self {
        Self {
            global_from_compact: new_storage(device, "global_from_compact", (n * 4) as u64),
            compact_from_global: new_storage(device, "compact_from_global", (n * 4) as u64),
            depths: new_storage(device, "depths", (n * 4) as u64),
            intersect_counts: new_storage(device, "intersect_counts", (n * 4) as u64),
            cum_tiles_hit: new_storage(device, "cum_tiles_hit", (n * 4) as u64),
            // Indexed by compact (visible) id, not by intersection.
            projected_splats: new_storage(device, "projected_splats", (n * 9 * 4) as u64),
            tile_id_from_isect: tile_pairs[0].keys.clone(),
            compact_sorted: tile_pairs[0].values.clone(),
            isect_id: tile_pairs[1].values.clone(),
            num_visible: new_storage(device, "num_visible", 4),
            num_intersections: new_storage(device, "num_intersections", 4),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            max_isects,
        }
    }
}

/// Read `num_visible` and `num_intersections` from their counter buffers.
fn read_counters(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    num_visible: &wgpu::Buffer,
    num_intersections: &wgpu::Buffer,
    readback: &wgpu::Buffer,
) -> (u32, u32) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(num_visible, 0, readback, 0, 4);
    encoder.copy_buffer_to_buffer(num_intersections, 0, readback, 4, 4);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let _ = rx.recv();
    let data = slice.get_mapped_range().expect("map range");
    let values: [u32; 2] = bytemuck::pod_read_unaligned(&data[..8]);
    drop(data);
    readback.unmap();
    (values[0], values[1])
}

const RO: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const RW: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };

// Bindings of the stages that touch intersection-sized buffers; shared by
// `Renderer::new` and `Renderer::ensure_isect_capacity`.
fn map_bindings(
    proj_uniforms: &wgpu::Buffer,
    gpu_scene: &GpuScene,
    scratch: &Scratch,
) -> Vec<Binding> {
    vec![
        Binding {
            binding: 0,
            ty: wgpu::BufferBindingType::Uniform,
            buffer: proj_uniforms.clone(),
        },
        Binding {
            binding: 1,
            ty: RO,
            buffer: gpu_scene.transforms.clone(),
        },
        Binding {
            binding: 2,
            ty: RO,
            buffer: gpu_scene.opacity.clone(),
        },
        Binding {
            binding: 3,
            ty: RO,
            buffer: scratch.cum_tiles_hit.clone(),
        },
        Binding {
            binding: 4,
            ty: RO,
            buffer: scratch.global_from_compact.clone(),
        },
        Binding {
            binding: 5,
            ty: RW,
            buffer: scratch.tile_id_from_isect.clone(),
        },
        Binding {
            binding: 6,
            ty: RW,
            buffer: scratch.compact_sorted.clone(),
        },
        Binding {
            binding: 7,
            ty: RW,
            buffer: scratch.isect_id.clone(),
        },
    ]
}

fn offsets_bindings(
    proj_uniforms: &wgpu::Buffer,
    scratch: &Scratch,
    tile_offsets: &wgpu::Buffer,
) -> Vec<Binding> {
    vec![
        Binding {
            binding: 0,
            ty: wgpu::BufferBindingType::Uniform,
            buffer: proj_uniforms.clone(),
        },
        Binding {
            binding: 1,
            ty: RO,
            buffer: scratch.tile_id_from_isect.clone(),
        },
        Binding {
            binding: 2,
            ty: RW,
            buffer: tile_offsets.clone(),
        },
    ]
}

fn raster_bindings(
    raster_uniforms: &wgpu::Buffer,
    scratch: &Scratch,
    tile_offsets: &wgpu::Buffer,
    out_img: &wgpu::Buffer,
    residuals: &PixelResiduals,
) -> Vec<Binding> {
    vec![
        Binding {
            binding: 0,
            ty: wgpu::BufferBindingType::Uniform,
            buffer: raster_uniforms.clone(),
        },
        Binding {
            binding: 1,
            ty: RO,
            buffer: scratch.projected_splats.clone(),
        },
        Binding {
            binding: 2,
            ty: RO,
            buffer: scratch.compact_sorted.clone(),
        },
        Binding {
            binding: 3,
            ty: RO,
            buffer: tile_offsets.clone(),
        },
        Binding {
            binding: 4,
            ty: RW,
            buffer: out_img.clone(),
        },
        Binding {
            binding: 5,
            ty: RO,
            buffer: scratch.global_from_compact.clone(),
        },
        Binding {
            binding: 6,
            ty: RW,
            buffer: residuals.final_t.clone(),
        },
        Binding {
            binding: 7,
            ty: RW,
            buffer: residuals.last_idx.clone(),
        },
    ]
}

/// Per-pixel forward residuals kept for the backward pass.
pub(crate) struct PixelResiduals {
    /// Transmittance left after the last blended splat.
    pub(crate) final_t: wgpu::Buffer,
    /// One past the tile-list index of the last blended splat (0 = none).
    pub(crate) last_idx: wgpu::Buffer,
}

/// The forward renderer.
pub struct Renderer {
    pub ctx: GpuContext,
    pub scene: GpuScene,
    scratch: Scratch,
    forward: Stage,
    visible: Stage,
    map: Stage,
    offsets: Stage,
    raster: Stage,
    proj_uniforms: wgpu::Buffer,
    raster_uniforms: wgpu::Buffer,
    tile_offsets: wgpu::Buffer,
    out_img: wgpu::Buffer,
    residuals: PixelResiduals,
    image_w: u32,
    image_h: u32,
    sh_degree: u32,
    // Device-side sorts and prefix scan.
    sorter: crate::sort::RadixSorter,
    scanner: crate::scan::PrefixScanner,
    depth_pairs: [crate::sort::SortBuffers; 2],
    tile_pairs: [crate::sort::SortBuffers; 2],
    counts_sorted: wgpu::Buffer,
    /// Hard ceiling for the isect buffers (device storage-binding limit).
    max_isects_limit: usize,
    /// Skip the output-image readback (for GPU-only timing / viewer use).
    skip_readback: bool,
    /// Evaluated SH degree override (see `set_active_sh_degree`).
    active_sh_degree: Option<u32>,
    /// The caller zeroes the parameter gradients after consuming them (e.g.
    /// a fused optimizer step), so `backward` need not clear them.
    pub(crate) grads_zeroed_by_caller: bool,
    /// Counts of the last rendered frame (what the backward pass differentiates).
    last_frame: FrameCounts,
    /// Backward-pass pipelines and buffers, built on first use.
    backward: Option<backward::BackwardState>,
}

/// Visible gaussians, intersections and tiles of a rendered frame.
#[derive(Clone, Copy, Default)]
struct FrameCounts {
    nv: usize,
    ni: usize,
    num_tiles: u32,
}

impl Renderer {
    pub fn new(
        ctx: GpuContext,
        scene: &Scene,
        image_w: u32,
        image_h: u32,
    ) -> Result<Self, GpuError> {
        Self::with_initial_isect_capacity(ctx, scene, image_w, image_h, None)
    }

    /// [`Renderer::new`] with an explicit initial intersection capacity
    /// (`None` = the default guess). Buffers still grow on demand; a small
    /// value lets tests exercise that growth cheaply.
    #[doc(hidden)]
    pub fn with_initial_isect_capacity(
        ctx: GpuContext,
        scene: &Scene,
        image_w: u32,
        image_h: u32,
        initial_isects: Option<usize>,
    ) -> Result<Self, GpuError> {
        let gpu_scene = GpuScene::upload(&ctx, scene);
        Self::build(ctx, gpu_scene, image_w, image_h, initial_isects)
    }

    /// [`Renderer::new`] from an already packed scene (no per-gaussian
    /// conversion; used by the trainer after densification).
    pub fn from_packed(
        ctx: GpuContext,
        packed: PackedScene,
        image_w: u32,
        image_h: u32,
    ) -> Result<Self, GpuError> {
        let gpu_scene = GpuScene::upload_packed(&ctx, packed);
        Self::build(ctx, gpu_scene, image_w, image_h, None)
    }

    /// [`Renderer::new`] over a scene whose buffers are already on the device
    /// (no upload).
    pub fn from_gpu_scene(
        ctx: GpuContext,
        gpu_scene: GpuScene,
        image_w: u32,
        image_h: u32,
    ) -> Result<Self, GpuError> {
        Self::build(ctx, gpu_scene, image_w, image_h, None)
    }

    fn build(
        ctx: GpuContext,
        gpu_scene: GpuScene,
        image_w: u32,
        image_h: u32,
        initial_isects: Option<usize>,
    ) -> Result<Self, GpuError> {
        let sh_degree = gpu_scene.packed.sh_degree;
        let n = gpu_scene.packed.num_gaussians;
        let (tbw, tbh) = tile_bounds(image_w, image_h);
        let num_tiles = (tbw * tbh) as usize;
        // Per-intersection buffers hold one u32 each. Start from a guess and
        // grow on demand (`ensure_isect_capacity`) up to the binding limit.
        let limit = ctx
            .limits
            .max_storage_buffer_binding_size
            .min(ctx.limits.max_buffer_size);
        let max_isects_limit = (limit / 4) as usize;
        let max_isects = initial_isects
            .unwrap_or_else(|| (n * 64).clamp(1 << 20, 1 << 26))
            .clamp(1, max_isects_limit);
        let dev = &ctx.device;
        // Device-side sort scratch. The depth sort and tile sort each need
        // ping-pong key/value pairs; the tile pairs double as the per-isect
        // scratch (see `Scratch`), so intersections cost 4 u32 buffers.
        let sort_capacity = n.max(max_isects).max(1);
        let sorter = crate::sort::RadixSorter::new(dev, sort_capacity);
        // The depth sort only ever holds the visible gaussians (<= n).
        let depth_pairs = sorter.allocate_len(dev, "depth", n.max(1));
        let tile_pairs = sorter.allocate_len(dev, "tile", max_isects);
        let scratch = Scratch::new(dev, n, max_isects, &tile_pairs);

        let proj_uniforms = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_uniforms"),
            size: std::mem::size_of::<ProjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_uniforms = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster_uniforms"),
            size: std::mem::size_of::<RasterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_offsets = new_storage(dev, "tile_offsets", (num_tiles * 2 * 4) as u64);
        let out_img = new_storage(dev, "out_img", (image_w as u64) * (image_h as u64) * 3 * 4);
        let pixels = image_w as u64 * image_h as u64;
        let residuals = PixelResiduals {
            final_t: new_storage(dev, "final_t", pixels * 4),
            last_idx: new_storage(dev, "last_idx", pixels * 4),
        };

        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        // Compact-order tile counts, gathered by `project_visible` for the scan.
        let counts_sorted = new_storage(dev, "counts_sorted", (n * 4) as u64);

        let forward = build_stage(
            dev,
            "project_forward",
            "project_forward",
            shaders::project_forward().source,
            &[
                Binding {
                    binding: 0,
                    ty: wgpu::BufferBindingType::Uniform,
                    buffer: proj_uniforms.clone(),
                },
                Binding {
                    binding: 1,
                    ty: ro,
                    buffer: gpu_scene.transforms.clone(),
                },
                Binding {
                    binding: 2,
                    ty: ro,
                    buffer: gpu_scene.opacity.clone(),
                },
                Binding {
                    binding: 3,
                    ty: rw,
                    buffer: scratch.global_from_compact.clone(),
                },
                Binding {
                    binding: 4,
                    ty: rw,
                    buffer: scratch.depths.clone(),
                },
                Binding {
                    binding: 5,
                    ty: rw,
                    buffer: scratch.intersect_counts.clone(),
                },
                Binding {
                    binding: 6,
                    ty: rw,
                    buffer: scratch.num_visible.clone(),
                },
                Binding {
                    binding: 7,
                    ty: rw,
                    buffer: scratch.num_intersections.clone(),
                },
            ],
        );

        let visible = build_stage(
            dev,
            "project_visible",
            "project_visible",
            shaders::project_visible().source,
            &[
                Binding {
                    binding: 0,
                    ty: wgpu::BufferBindingType::Uniform,
                    buffer: proj_uniforms.clone(),
                },
                Binding {
                    binding: 1,
                    ty: ro,
                    buffer: gpu_scene.transforms.clone(),
                },
                Binding {
                    binding: 2,
                    ty: ro,
                    buffer: gpu_scene.opacity.clone(),
                },
                Binding {
                    binding: 3,
                    ty: ro,
                    buffer: gpu_scene.sh.clone(),
                },
                Binding {
                    binding: 4,
                    ty: ro,
                    buffer: scratch.global_from_compact.clone(),
                },
                Binding {
                    binding: 5,
                    ty: rw,
                    buffer: scratch.projected_splats.clone(),
                },
                Binding {
                    binding: 6,
                    ty: rw,
                    buffer: scratch.compact_from_global.clone(),
                },
                Binding {
                    binding: 7,
                    ty: ro,
                    buffer: scratch.intersect_counts.clone(),
                },
                Binding {
                    binding: 8,
                    ty: rw,
                    buffer: counts_sorted.clone(),
                },
            ],
        );

        let map = build_stage(
            dev,
            "map_gaussians",
            "map_gaussians",
            shaders::map_gaussians().source,
            &map_bindings(&proj_uniforms, &gpu_scene, &scratch),
        );

        let offsets = build_stage(
            dev,
            "get_tile_offsets",
            "get_tile_offsets",
            shaders::tile_offsets().source,
            &offsets_bindings(&proj_uniforms, &scratch, &tile_offsets),
        );

        let raster = build_stage(
            dev,
            "rasterize",
            "rasterize",
            shaders::rasterize().source,
            &raster_bindings(
                &raster_uniforms,
                &scratch,
                &tile_offsets,
                &out_img,
                &residuals,
            ),
        );

        let scanner = crate::scan::PrefixScanner::new(dev, n.max(1));

        Ok(Self {
            ctx,
            scene: gpu_scene,
            scratch,
            forward,
            visible,
            map,
            offsets,
            raster,
            proj_uniforms,
            raster_uniforms,
            tile_offsets,
            out_img,
            residuals,
            image_w,
            image_h,
            sh_degree,
            sorter,
            scanner,
            depth_pairs,
            tile_pairs,
            counts_sorted,
            max_isects_limit,
            skip_readback: false,
            active_sh_degree: None,
            grads_zeroed_by_caller: false,
            last_frame: FrameCounts::default(),
            backward: None,
        })
    }

    /// Grow every intersection-sized buffer (and the tile sort scratch) so a
    /// frame with `needed` intersections fits. Views from inside a scene can
    /// produce far more intersections than the initial guess (e.g. 31M at
    /// 1080p for 419k gaussians); without this they were truncated.
    fn ensure_isect_capacity(&mut self, needed: usize) {
        if needed <= self.scratch.max_isects || self.scratch.max_isects >= self.max_isects_limit {
            return;
        }
        let cap = needed.saturating_add(needed / 4).min(self.max_isects_limit);
        let dev = &self.ctx.device;
        self.sorter.reserve(dev, cap);
        self.tile_pairs = self.sorter.allocate_len(dev, "tile", cap);
        self.scratch.tile_id_from_isect = self.tile_pairs[0].keys.clone();
        self.scratch.compact_sorted = self.tile_pairs[0].values.clone();
        self.scratch.isect_id = self.tile_pairs[1].values.clone();
        self.scratch.max_isects = cap;
        self.map.rebind(
            dev,
            "map_gaussians",
            &map_bindings(&self.proj_uniforms, &self.scene, &self.scratch),
        );
        self.offsets.rebind(
            dev,
            "get_tile_offsets",
            &offsets_bindings(&self.proj_uniforms, &self.scratch, &self.tile_offsets),
        );
        self.raster.rebind(
            dev,
            "rasterize",
            &raster_bindings(
                &self.raster_uniforms,
                &self.scratch,
                &self.tile_offsets,
                &self.out_img,
                &self.residuals,
            ),
        );
    }

    /// Render without reading the output image back to the CPU.
    ///
    /// A native viewer presents the output buffer directly, so this is the
    /// per-frame cost that matters; use it to measure the true GPU frame time.
    /// Promise that the parameter gradient buffers are zero at the start of
    /// every backward pass (the caller clears them after reading, as the
    /// trainer's Adam step does), which saves a full clear per step.
    pub fn set_grads_zeroed_by_caller(&mut self, zeroed: bool) {
        self.grads_zeroed_by_caller = zeroed;
    }

    /// Evaluate spherical harmonics only up to `degree` (clamped to the
    /// scene's degree); higher bands contribute nothing and get zero
    /// gradients. `None` evaluates the full degree. Trainers use this for
    /// the usual one-band-per-1000-steps schedule.
    pub fn set_active_sh_degree(&mut self, degree: Option<u32>) {
        self.active_sh_degree = degree;
    }

    pub fn set_skip_readback(&mut self, skip: bool) {
        self.skip_readback = skip;
    }

    /// Give the GPU context back (e.g. to rebuild the renderer for a scene
    /// with a different number of gaussians).
    pub fn into_context(self) -> GpuContext {
        self.ctx
    }

    pub fn num_gaussians(&self) -> usize {
        self.scene.packed.num_gaussians
    }

    /// Render `view` and return the linear-RGB image.
    pub fn render(&mut self, view: &CameraView, bg: [f32; 3]) -> Image {
        assert_eq!(
            view.camera.width, self.image_w,
            "view width matches renderer"
        );
        assert_eq!(
            view.camera.height, self.image_h,
            "view height matches renderer"
        );
        let mut prof = StageTimer::from_env();
        let mut u = ProjectUniforms::from_view(view, self.sh_degree, self.num_gaussians() as u32);
        u.sh_active_degree = self
            .active_sh_degree
            .map_or(self.sh_degree, |d| d.min(self.sh_degree));
        let num_tiles = u.num_tiles();
        let raster_u = RasterUniforms::new(&u, bg);
        let n = self.num_gaussians() as u32;

        // ----- Upload uniforms + reset dynamic buffers. -----
        self.ctx
            .queue
            .write_buffer(&self.proj_uniforms, 0, bytemuck::bytes_of(&u));
        self.ctx
            .queue
            .write_buffer(&self.raster_uniforms, 0, bytemuck::bytes_of(&raster_u));
        self.ctx
            .queue
            .write_buffer(&self.scratch.num_visible, 0, &[0u8; 4]);
        self.ctx
            .queue
            .write_buffer(&self.scratch.num_intersections, 0, &[0u8; 4]);
        let mut init = vec![0u32; num_tiles as usize * 2];
        for t in 0..num_tiles as usize {
            init[t * 2] = u32::MAX;
        }
        self.ctx
            .queue
            .write_buffer(&self.tile_offsets, 0, bytemuck::cast_slice(&init));

        // ----- Pass 1: project_forward. -----
        {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pf") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                dispatch(&mut pass, &self.forward, n.div_ceil(256));
            }
            self.ctx.queue.submit(Some(encoder.finish()));
        }
        prof.mark(&self.ctx.device, "project_fwd");

        // ----- Host readback of compaction counts. -----
        let (num_visible, num_intersections) = read_counters(
            &self.ctx.device,
            &self.ctx.queue,
            &self.scratch.num_visible,
            &self.scratch.num_intersections,
            &self.scratch.readback,
        );
        prof.mark(&self.ctx.device, "counters");
        if prof.enabled() {
            eprintln!(
                "[profile] counters raw: visible={num_visible} isects={num_intersections} (cap {})",
                self.scratch.max_isects
            );
        }
        self.ensure_isect_capacity(num_intersections as usize);
        let nv = (num_visible as usize).min(self.num_gaussians());
        // Only truncates if the device's storage-binding limit is exceeded.
        let ni = (num_intersections as usize).min(self.scratch.max_isects);
        self.last_frame = FrameCounts { nv, ni, num_tiles };

        // Re-upload the uniforms with the now-known compaction counts.
        let u = ProjectUniforms {
            num_visible: nv as u32,
            num_intersections: ni as u32,
            ..u
        };
        self.ctx
            .queue
            .write_buffer(&self.proj_uniforms, 0, bytemuck::bytes_of(&u));

        // ----- Device-side depth sort. -----
        //
        // Keys are the raw f32 depth bits (monotonic for positive z), values are
        // the global gaussian ids, so ascending key order is nearest-first
        // (front-to-back). Full 32-bit keys: eight ping-pong passes.
        if nv > 0 {
            self.sort_in_place(
                &self.scratch.depths,
                &self.scratch.global_from_compact,
                &self.depth_pairs,
                nv,
                32,
            );
        }
        prof.mark(&self.ctx.device, "depth_sort");

        // ----- Pass 2a: project_visible (also gathers tile counts). -----
        if nv > 0 {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pv") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                dispatch(&mut pass, &self.visible, (nv as u32).div_ceil(256));
            }
            self.ctx.queue.submit(Some(encoder.finish()));
        }
        prof.mark(&self.ctx.device, "project_vis");

        // ----- Device-side prefix scan of the compact-order tile counts. -----
        if nv > 0 {
            self.scanner.scan(
                &self.ctx.device,
                &self.ctx.queue,
                &self.counts_sorted,
                &self.scratch.cum_tiles_hit,
                nv,
            );
        }
        prof.mark(&self.ctx.device, "scan");

        // ----- Pass 2b: map_gaussians. -----
        if nv > 0 {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("map") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                dispatch(&mut pass, &self.map, (nv as u32).div_ceil(256));
            }
            self.ctx.queue.submit(Some(encoder.finish()));
        }
        prof.mark(&self.ctx.device, "map");

        // ----- Device-side tile-id sort over the isect list. -----
        if ni > 0 {
            // Sort (tile id, compact id) pairs in `tile_pairs` directly. Keys
            // are tile ids < num_tiles; the LSD sort is stable, so the depth
            // order from the first sort survives within each tile.
            let key_bits = crate::sort::key_bits_for(num_tiles.saturating_sub(1));
            self.sorter.prepare(&self.ctx.queue, ni, key_bits);
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("tile_sort"),
                    });
            let out = self.sorter.encode(
                &self.ctx.device,
                &mut encoder,
                &self.tile_pairs,
                ni,
                key_bits,
            );
            if out == 1 {
                // Downstream stages read pair 0.
                let len = (ni * 4) as u64;
                let [a, b] = &self.tile_pairs;
                encoder.copy_buffer_to_buffer(&b.keys, 0, &a.keys, 0, len);
                encoder.copy_buffer_to_buffer(&b.values, 0, &a.values, 0, len);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
        }
        prof.mark(&self.ctx.device, "tile_sort");

        // ----- Pass 3: tile_offsets + rasterize. -----
        {
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("raster"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                dispatch_threads(&mut pass, &self.offsets, ni as u32);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
        }
        prof.mark(&self.ctx.device, "tile_offsets");
        {
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("raster"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                dispatch(&mut pass, &self.raster, num_tiles);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
        }
        prof.mark(&self.ctx.device, "rasterize");
        if prof.enabled() {
            prof.report(nv, ni, num_tiles);
            eprintln!("[profile] {}", self.tile_list_stats(num_tiles));
        }

        // A viewer renders to a surface and never reads the image back; this
        // path exists so the true GPU frame cost can be measured.
        if self.skip_readback {
            return Image {
                width: self.image_w,
                height: self.image_h,
                rgb: Vec::new(),
            };
        }

        let t_read = std::time::Instant::now();
        let rgb = read_f32s(
            &self.ctx.device,
            &self.ctx.queue,
            &self.out_img,
            (self.image_w as usize) * (self.image_h as usize) * 3,
        );
        if prof.enabled() {
            eprintln!("[profile] image readback: {:?}", t_read.elapsed());
        }
        let rgb = rgb.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Image {
            width: self.image_w,
            height: self.image_h,
            rgb,
        }
    }

    /// Per-tile isect list length statistics (profiling only): a long tail
    /// means `rasterize` (one workgroup per tile) is load-imbalanced.
    fn tile_list_stats(&self, num_tiles: u32) -> String {
        let raw = read_bytes(
            &self.ctx.device,
            &self.ctx.queue,
            &self.tile_offsets,
            num_tiles as usize * 2 * 4,
        );
        let ranges: &[u32] = bytemuck::cast_slice(&raw);
        let mut lens: Vec<u32> = ranges
            .chunks_exact(2)
            .map(|r| if r[0] == u32::MAX { 0 } else { r[1] - r[0] })
            .collect();
        lens.sort_unstable();
        let total: u64 = lens.iter().map(|&l| l as u64).sum();
        let pct = |q: f64| lens[((lens.len() - 1) as f64 * q) as usize];
        format!(
            "tile lists: tiles={} mean={:.0} p50={} p90={} p99={} max={} (max/mean={:.1})",
            lens.len(),
            total as f64 / lens.len() as f64,
            pct(0.5),
            pct(0.9),
            pct(0.99),
            pct(1.0),
            pct(1.0) as f64 / (total as f64 / lens.len() as f64).max(1.0)
        )
    }

    /// Sort `n` (key, value) pairs held in `keys`/`values` in place, using
    /// `pairs` as ping-pong scratch; keys must fit in `key_bits` bits. Copy-in,
    /// the radix passes and copy-out are recorded into a single command buffer
    /// (one submit).
    fn sort_in_place(
        &self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        pairs: &[crate::sort::SortBuffers; 2],
        n: usize,
        key_bits: u32,
    ) {
        let len = (n * 4) as u64;
        self.sorter.prepare(&self.ctx.queue, n, key_bits);
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sort"),
            });
        encoder.copy_buffer_to_buffer(keys, 0, &pairs[0].keys, 0, len);
        encoder.copy_buffer_to_buffer(values, 0, &pairs[0].values, 0, len);
        let out = self
            .sorter
            .encode(&self.ctx.device, &mut encoder, pairs, n, key_bits);
        encoder.copy_buffer_to_buffer(&pairs[out].keys, 0, keys, 0, len);
        encoder.copy_buffer_to_buffer(&pairs[out].values, 0, values, 0, len);
        self.ctx.queue.submit(Some(encoder.finish()));
    }
}

fn read_f32s(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    let raw = read_bytes(device, queue, buffer, count * 4);
    bytemuck::cast_slice(&raw).to_vec()
}

fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("staging-copy"),
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

/// Opt-in (`GSPLAT_PROFILE`) per-stage wall-clock timer. Each `mark` blocks
/// until the GPU is idle, so stages are serialised and their costs separable;
/// when disabled it is a no-op and the frame stays pipelined.
struct StageTimer {
    last: Option<std::time::Instant>,
    stages: Vec<(&'static str, std::time::Duration)>,
}

impl StageTimer {
    fn from_env() -> Self {
        Self {
            last: std::env::var("GSPLAT_PROFILE")
                .is_ok()
                .then(std::time::Instant::now),
            stages: Vec::new(),
        }
    }

    fn enabled(&self) -> bool {
        self.last.is_some()
    }

    fn mark(&mut self, device: &wgpu::Device, stage: &'static str) {
        let Some(last) = self.last else {
            return;
        };
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let now = std::time::Instant::now();
        self.stages.push((stage, now - last));
        self.last = Some(now);
    }

    fn report(&self, nv: usize, ni: usize, num_tiles: u32) {
        let total: std::time::Duration = self.stages.iter().map(|s| s.1).sum();
        let parts: Vec<String> = self
            .stages
            .iter()
            .map(|(name, d)| format!("{name}={:.2}", d.as_secs_f64() * 1e3))
            .collect();
        eprintln!(
            "[profile] ms: {} total={:.2} | visible={nv} isects={ni} tiles={num_tiles}",
            parts.join(" "),
            total.as_secs_f64() * 1e3
        );
    }
}

#[path = "renderer_backward.rs"]
mod backward;
pub use backward::{DeviceParams, ParamGrads, SCREEN_GRAD_FLOATS};
