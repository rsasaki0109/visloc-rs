//! PatchMatch on the GPU (wgpu): [`crate::patch_match`]'s per-pixel update
//! as compute shaders (`shaders/patch_match.wgsl`), one invocation per
//! pixel, in f32. It follows the CPU path step for step (sampling regions,
//! view selection, refinement and their random numbers), so the two differ
//! only by floating-point rounding.

use visloc_gsplat_render::{GpuContext, GpuError};

use crate::deformable::{self, Class, MAX_ANCHORS};
use crate::pass_setup::{Params, PassSetup};
use crate::patch_match::{
    finish_pass_depth_map, DepthMap, DetailRestorer, PatchMatchOptions, MAX_COST,
};
use crate::planar_prior::{PlanarPrior, PlanarPriorOptions};
use crate::workspace::Workspace;

/// Uniform slots are this far apart (wgpu's minimum offset alignment).
const SLOT: u64 = 256;

/// The GPU PatchMatch: a device and the compute pipelines.
pub struct GpuPatchMatch {
    ctx: GpuContext,
    init: wgpu::ComputePipeline,
    propagate: wgpu::ComputePipeline,
    classify: wgpu::ComputePipeline,
    find_anchors: wgpu::ComputePipeline,
}

/// A GPU pass's planes, costs, the pixels sorted for the next pass (empty
/// unless asked for) and, in a deformable pass, the classes it ran with.
type PassOutput = (Vec<[f32; 4]>, Vec<f32>, Vec<Class>, Option<Vec<Class>>);

impl GpuPatchMatch {
    /// Creates a device (`GpuContext::new`) and compiles the shaders.
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self::with_context(GpuContext::new()?))
    }

    /// Uses an existing device.
    pub fn with_context(ctx: GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mvs-patch-match"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/patch_match.wgsl").into()),
            });
        let pipeline = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        };
        let init = pipeline("init");
        let propagate = pipeline("propagate");
        let classify = pipeline("classify");
        let find_anchors = pipeline("find_anchors");
        Self {
            ctx,
            init,
            propagate,
            classify,
            find_anchors,
        }
    }

    /// The adapter's name and backend, for logs.
    pub fn adapter(&self) -> String {
        let info = &self.ctx.adapter_info;
        format!("{} ({:?})", info.name, info.backend)
    }

    /// [`crate::patch_match::estimate`] on the GPU.
    pub fn estimate(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        prior: Option<&[DepthMap]>,
    ) -> DepthMap {
        let start = prior.map(|maps| &maps[view]);
        let (planes, costs, classes, pass_classes) =
            self.run(ws, view, options, prior, start, None, DetailRestorer::Off);
        DepthMap {
            classes,
            ..finish_pass_depth_map(
                &ws.views[view],
                &planes,
                costs,
                options,
                pass_classes.as_deref(),
            )
        }
    }

    /// [`crate::patch_match::estimate_from`] on the GPU.
    pub fn estimate_from(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        start: &DepthMap,
        restorer: DetailRestorer,
    ) -> DepthMap {
        let (planes, costs, classes, pass_classes) =
            self.run(ws, view, options, None, Some(start), None, restorer);
        DepthMap {
            classes,
            ..finish_pass_depth_map(
                &ws.views[view],
                &planes,
                costs,
                options,
                pass_classes.as_deref(),
            )
        }
    }

    /// [`crate::patch_match::estimate_with_prior`] on the GPU.
    pub fn estimate_with_prior(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        prior_options: &PlanarPriorOptions,
        start: &DepthMap,
        prior: &PlanarPrior,
    ) -> DepthMap {
        let (planes, costs, classes, pass_classes) = self.run(
            ws,
            view,
            options,
            None,
            Some(start),
            Some((prior_options, prior)),
            DetailRestorer::Off,
        );
        DepthMap {
            classes,
            ..finish_pass_depth_map(
                &ws.views[view],
                &planes,
                costs,
                options,
                pass_classes.as_deref(),
            )
        }
    }

    /// [`crate::patch_match::photometric_score`] on the GPU: the first
    /// pass alone, started from `map`.
    pub fn photometric_score(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        map: &DepthMap,
    ) -> Vec<f32> {
        let first_pass = PatchMatchOptions {
            iterations: 0,
            sort_pixels: None,
            ..options.clone()
        };
        let (_, mut costs, _, _) = self.run(
            ws,
            view,
            &first_pass,
            None,
            Some(map),
            None,
            DetailRestorer::Off,
        );
        for (cost, &depth) in costs.iter_mut().zip(&map.depth) {
            if depth <= 0.0 {
                *cost = MAX_COST;
            }
        }
        costs
    }

    /// The per-pixel planes and costs after `options.iterations`
    /// iterations (and the pixels sorted, with `options.sort_pixels`),
    /// starting from `start`'s planes (random ones where it
    /// has no depth, or everywhere without it), with geometric consistency
    /// against `prior`, or as the planar prior pass with `planar`'s prior
    /// (where `start`'s cost is not credible, from the prior's plane), and
    /// `restorer` deciding which pixels may leave their start plane.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        prior: Option<&[DepthMap]>,
        start: Option<&DepthMap>,
        planar: Option<(&PlanarPriorOptions, &PlanarPrior)>,
        restorer: DetailRestorer,
    ) -> PassOutput {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let reference = &ws.views[view];
        let PassSetup {
            w,
            h,
            source_ids,
            sources,
            texture_width,
            texture_height,
            layers,
            prior_len,
            init_planes,
            base,
            propagate,
            deformable,
            sort,
        } = PassSetup::new(ws, view, options, prior, start, planar, restorer);
        // The texture holds half floats (the f32 layers are freed here).
        let layers: Vec<u16> = layers.into_iter().map(f16_bits).collect();
        let deformable = deformable.as_ref();
        // Slot 0 for `init`, then one per (iteration, colour), then
        // `classify` and `find_anchors` at the start, then `classify` at
        // the end.
        let slots = 1 + propagate.len();
        let anchor_slot = slots;
        let sort_slot = slots + 1;
        let mut uniform_bytes = vec![0u8; (slots + 2) * SLOT as usize];
        let mut put = |slot: usize, params: Params| {
            let bytes = bytemuck::bytes_of(&params);
            uniform_bytes[slot * SLOT as usize..slot * SLOT as usize + bytes.len()]
                .copy_from_slice(bytes);
        };
        put(0, base);
        put(anchor_slot, base);
        if let Some((_, params)) = &sort {
            put(sort_slot, *params);
        }
        for (k, params) in propagate.iter().enumerate() {
            put(1 + k, *params);
        }
        let sort = sort.as_ref().map(|(d, _)| d);

        // wgpu zero-initializes buffers, so only data is uploaded.
        let zeroed = |label: &str, len: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len as u64).max(16).div_ceil(16) * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let storage = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            let buffer = zeroed(label, bytes.len());
            if !bytes.is_empty() {
                queue.write_buffer(&buffer, 0, bytes);
            }
            buffer
        };
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mvs-params"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms, 0, &uniform_bytes);
        let sources_buf = storage("mvs-sources", bytemuck::cast_slice(&sources));
        let images_buf = storage("mvs-images", bytemuck::cast_slice(&reference.gray_levels()));
        let priors_buf = zeroed("mvs-priors", prior_len * 4);
        if let Some(maps) = prior {
            for (&s, source) in source_ids.iter().zip(&sources) {
                queue.write_buffer(
                    &priors_buf,
                    u64::from(source.dims[3]) * 4,
                    bytemuck::cast_slice(&maps[s].depth),
                );
            }
        }
        let planes_buf = zeroed("mvs-planes", w * h * 16);
        let costs_buf = zeroed("mvs-costs", w * h * 4);
        let best_buf = zeroed("mvs-best-views", w * h * 4);
        let init_buf = storage("mvs-init-planes", bytemuck::cast_slice(&init_planes));
        let prior_planes_buf = match planar {
            Some((_, planar)) => storage("mvs-prior-planes", bytemuck::cast_slice(&planar.planes)),
            None => zeroed("mvs-prior-planes", 16),
        };
        let pre_costs_buf = match restorer {
            DetailRestorer::Off => zeroed("mvs-pre-costs", 16),
            _ => zeroed("mvs-pre-costs", w * h * 4),
        };
        let per_pixel = |used: bool, bytes: usize| if used { w * h * bytes } else { 16 };
        let classes_buf = zeroed(
            "mvs-classes",
            per_pixel(deformable.is_some() || sort.is_some(), 4),
        );
        let nearest_buf = zeroed("mvs-nearest", per_pixel(deformable.is_some(), 4));
        let anchors_buf = zeroed(
            "mvs-anchors",
            per_pixel(deformable.is_some(), 4 * MAX_ANCHORS),
        );
        let texture_size = wgpu::Extent3d {
            width: texture_width as u32,
            height: texture_height as u32,
            depth_or_array_layers: source_ids.len().max(1) as u32,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mvs-source-images"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&layers),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * texture_width as u32),
                rows_per_image: Some(texture_height as u32),
            },
            texture_size,
        );
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mvs-bilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        drop(layers);
        drop(init_planes);

        let params_entry = |slot: usize| wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniforms,
                offset: slot as u64 * SLOT,
                size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
            }),
        };
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let init_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mvs-init"),
            layout: &self.init.get_bind_group_layout(0),
            entries: &[
                params_entry(0),
                entry(1, &sources_buf),
                entry(2, &images_buf),
                entry(3, &priors_buf),
                entry(4, &planes_buf),
                entry(5, &costs_buf),
                entry(6, &best_buf),
                entry(7, &init_buf),
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                entry(11, &pre_costs_buf),
            ],
        });
        let propagate_groups: Vec<wgpu::BindGroup> = (1..slots)
            .map(|slot| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mvs-propagate"),
                    layout: &self.propagate.get_bind_group_layout(0),
                    entries: &[
                        params_entry(slot),
                        entry(1, &sources_buf),
                        entry(2, &images_buf),
                        entry(3, &priors_buf),
                        entry(4, &planes_buf),
                        entry(5, &costs_buf),
                        entry(6, &best_buf),
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        entry(10, &prior_planes_buf),
                        entry(11, &pre_costs_buf),
                        entry(14, &anchors_buf),
                    ],
                })
            })
            .collect();

        // One submission per dispatch keeps each well under any watchdog.
        let run =
            |pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup, groups: (u32, u32)| {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mvs-dispatch"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("mvs-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
                queue.submit(Some(encoder.finish()));
            };
        let (gw, gh) = ((w as u32).div_ceil(16), (h as u32).div_ceil(8));
        let debug = std::env::var_os("VISLOC_MVS_DEBUG").is_some();
        let mut anchored: Option<Vec<bool>> = None;
        run(&self.init, &init_group, (gw, gh));
        // `classify` on the current planes with the settings in `slot`.
        let classify = |slot: usize| -> Vec<Class> {
            let classify_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mvs-classify"),
                layout: &self.classify.get_bind_group_layout(0),
                entries: &[
                    params_entry(slot),
                    entry(1, &sources_buf),
                    entry(2, &images_buf),
                    entry(3, &priors_buf),
                    entry(4, &planes_buf),
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    entry(12, &classes_buf),
                ],
            });
            run(&self.classify, &classify_group, (gw, gh));
            let mut codes = vec![0u32; w * h];
            read_into(device, queue, &classes_buf, &mut codes);
            codes
                .iter()
                .map(|&c| match c {
                    0 => Class::Weak,
                    1 => Class::Strong,
                    _ => Class::Unknown,
                })
                .collect()
        };
        let mut pass_classes = None;
        if let Some(d) = deformable {
            // The start's classes if the pass that made it sorted them,
            // else the start's pixels sorted now; each pixel's nearest
            // strong pixel on the CPU (`deformable::nearest_strong`), then
            // the anchors.
            let start_classes = start.map_or(&[][..], |map| &map.classes[..]);
            let classes = if start_classes.len() == w * h {
                let codes: Vec<u32> = start_classes
                    .iter()
                    .map(|c| match c {
                        Class::Weak => 0,
                        Class::Strong => 1,
                        Class::Unknown => 2,
                    })
                    .collect();
                queue.write_buffer(&classes_buf, 0, bytemuck::cast_slice(&codes));
                start_classes.to_vec()
            } else {
                classify(anchor_slot)
            };
            let nearest = deformable::nearest_strong(&classes, w, h, d.nearest_radius);
            pass_classes = Some(classes.clone());
            queue.write_buffer(&nearest_buf, 0, bytemuck::cast_slice(&nearest));
            let anchors_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mvs-find-anchors"),
                layout: &self.find_anchors.get_bind_group_layout(0),
                entries: &[
                    params_entry(anchor_slot),
                    entry(4, &planes_buf),
                    entry(12, &classes_buf),
                    entry(13, &nearest_buf),
                    entry(14, &anchors_buf),
                ],
            });
            run(&self.find_anchors, &anchors_group, (gw, gh));
            if debug {
                let mut codes = vec![0u32; w * h];
                read_into(device, queue, &classes_buf, &mut codes);
                let share = |c: u32| {
                    100.0 * codes.iter().filter(|&&x| x == c).count() as f64 / (w * h) as f64
                };
                eprintln!(
                    "DEBUG view {view} {w}x{h} geometric {} prior {} restore {}: weak {:.1}% (anchored {:.1}%) strong {:.1}% unknown {:.1}%",
                    prior.is_some(),
                    planar.is_some(),
                    base.restore,
                    share(0) + share(3),
                    share(3),
                    share(1),
                    share(2)
                );
                anchored = Some(codes.iter().map(|&c| c == 3).collect());
            }
        }
        let half = (w as u32).div_ceil(2).div_ceil(16);
        for group in &propagate_groups {
            run(&self.propagate, group, (half, gh));
        }

        let classes = if sort.is_some() {
            classify(sort_slot)
        } else {
            Vec::new()
        };
        let mut planes = vec![[0.0f32; 4]; w * h];
        read_into(device, queue, &planes_buf, &mut planes);
        let mut costs = vec![0.0f32; w * h];
        read_into(device, queue, &costs_buf, &mut costs);
        if let (Some(anchored), Some(start)) = (&anchored, start) {
            // Pixels whose depth moved by more than 1% from the start, with
            // and without anchors.
            let (fx, fy) = (reference.fx as f32, reference.fy as f32);
            let (cx, cy) = (reference.cx as f32, reference.cy as f32);
            let (mut moved, mut total) = ([0usize; 2], [0usize; 2]);
            for (i, p) in planes.iter().enumerate() {
                let d0 = start.depth[i];
                if d0 <= 0.0 {
                    continue;
                }
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let denom = p[0] * (x - cx) / fx + p[1] * (y - cy) / fy + p[2];
                let d = if denom < -1e-6 { p[3] / denom } else { 0.0 };
                let k = usize::from(anchored[i]);
                total[k] += 1;
                if d <= 0.0 || d.is_nan() || ((d - d0) / d0).abs() > 0.01 {
                    moved[k] += 1;
                }
            }
            let pct = |k: usize| 100.0 * moved[k] as f64 / total[k].max(1) as f64;
            eprintln!(
                "DEBUG view {view}: moved >1%: anchored {:.1}% of {}, others {:.1}% of {}",
                pct(1),
                total[1],
                pct(0),
                total[0]
            );
        }
        (planes, costs, classes, pass_classes)
    }
}

/// `x` as an IEEE half float, rounded to nearest even.
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity, or a quiet NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero): the implicit bit joins the mantissa.
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = m + half - 1 + ((m >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    let mut out = ((e as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && out & 1 == 1) {
        // A carry into the exponent is still the right rounding.
        out += 1;
    }
    sign | out as u16
}

/// Copies the start of `buffer` back to the host, into `out`.
fn read_into<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    out: &mut [T],
) {
    let len = std::mem::size_of_val(out);
    if len == 0 {
        return;
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mvs-staging"),
        size: len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mvs-readback"),
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
    bytemuck::cast_slice_mut::<T, u8>(out)
        .copy_from_slice(&slice.get_mapped_range().expect("map range"));
    staging.unmap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_setup::checks;

    checks::impl_device!(GpuPatchMatch);

    /// A device on the first adapter; the tests skip without one.
    fn gpu() -> Option<GpuPatchMatch> {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("no GPU adapter; skipping");
            return None;
        };
        Some(GpuPatchMatch::with_context(ctx))
    }

    #[test]
    fn gpu_prior_fills_a_textureless_region() {
        if let Some(gpu) = gpu() {
            checks::prior_fills_a_textureless_region(&gpu);
        }
    }

    #[test]
    fn gpu_patch_match_recovers_a_plane() {
        if let Some(gpu) = gpu() {
            checks::recovers_a_plane(&gpu);
        }
    }

    #[test]
    fn gpu_restorer_matches_the_cpu() {
        if let Some(gpu) = gpu() {
            checks::restorer_matches_the_cpu(&gpu);
        }
    }

    #[test]
    fn gpu_deformable_patches_fill_a_textureless_region() {
        if let Some(gpu) = gpu() {
            checks::deformable_patches_fill_a_textureless_region(&gpu);
        }
    }

    #[test]
    fn gpu_strong_median_follows_the_pass_classes() {
        if let Some(gpu) = gpu() {
            checks::strong_median_follows_the_pass_classes(&gpu);
        }
    }

    #[test]
    fn gpu_deformable_passes_take_the_start_classes() {
        if let Some(gpu) = gpu() {
            checks::deformable_passes_take_the_start_classes(&gpu);
        }
    }

    #[test]
    fn half_floats_round_to_nearest_even() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(2.0), 0x4000);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(255.0), 0x5bf8);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1e6), 0x7c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        // One ulp up, then halfway cases to the even neighbour.
        assert_eq!(f16_bits(1.0 + 1.0 / 1024.0), 0x3c01);
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        // The smallest subnormal, and half of it to zero (even).
        assert_eq!(f16_bits(2f32.powi(-24)), 1);
        assert_eq!(f16_bits(2f32.powi(-25)), 0);
        assert_eq!(f16_bits(3.0 * 2f32.powi(-25)), 2);
    }

    /// The GPU's photometric score of a map matches the CPU's, up to the
    /// texture unit's half floats and interpolation weights.
    #[test]
    fn gpu_photometric_score_matches_the_cpu() {
        if let Some(gpu) = gpu() {
            checks::photometric_score_matches_the_cpu(&gpu, 1e-3, 1e-2);
        }
    }
}
