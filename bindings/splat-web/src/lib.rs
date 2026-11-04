//! visloc-rs's 3D Gaussian Splatting renderer ([`visloc_gsplat_render`]) in
//! the browser: the page hands it a scene (`.splat` bytes) and a camera, it
//! renders with WebGPU through wgpu and draws the frame into a canvas without
//! the image leaving the GPU ([`Renderer::render_async`], then a full-screen
//! pass that reads [`Renderer::output`]).
//!
//! The camera math and the presenting pass build natively too, so they are
//! tested without a browser; the canvas part ([`SplatViewer`]) is wasm only.

use nalgebra::{Matrix3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::Scene;
#[cfg(target_arch = "wasm32")]
use visloc_gsplat_render::{GpuContext, Renderer};

/// Background of the frame (the site's page colour; splat colours are display
/// values, as the training images were).
pub const BACKGROUND: [f32; 3] = [0.055, 0.067, 0.086];

/// A look-at view in the renderer's convention (world to camera; x right,
/// y down, z forward): a pinhole camera of `width` x `height` pixels with the
/// vertical field of view `fov_y` (radians) at `eye`, looking at `target`,
/// `up` the world's up direction.
pub fn look_at(
    eye: Vector3<f32>,
    target: Vector3<f32>,
    up: Vector3<f32>,
    fov_y: f32,
    width: u32,
    height: u32,
) -> CameraView {
    let forward = (target - eye).normalize();
    let right = forward.cross(&up).normalize();
    let down = forward.cross(&right);
    let rotation = Matrix3::from_rows(&[right.transpose(), down.transpose(), forward.transpose()]);
    let f = 0.5 * height as f32 / (0.5 * fov_y).tan();
    CameraView::new(
        rotation,
        -(rotation * eye),
        PinholeCamera::new(width, height, f, f, 0.5 * width as f32, 0.5 * height as f32),
    )
}

/// A robust centre and size of a scene for the first view: the per-axis
/// median of the gaussians' means, and the 90th percentile of their distances
/// to it.
pub fn scene_bounds(scene: &Scene) -> ([f32; 3], f32) {
    if scene.gaussians.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let median = |axis: usize| {
        let mut v: Vec<f32> = scene.gaussians.iter().map(|g| g.mean[axis]).collect();
        let mid = v.len() / 2;
        *v.select_nth_unstable_by(mid, f32::total_cmp).1
    };
    let centre = Vector3::new(median(0), median(1), median(2));
    let mut dist: Vec<f32> = scene
        .gaussians
        .iter()
        .map(|g| (g.mean - centre).norm())
        .collect();
    let k = (dist.len() * 9 / 10).min(dist.len() - 1);
    let radius = *dist.select_nth_unstable_by(k, f32::total_cmp).1;
    ([centre.x, centre.y, centre.z], radius.max(1e-3))
}

const PRESENT_WGSL: &str = r#"
struct Dims { width: u32, height: u32, srgb_target: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<uniform> dims: Dims;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // one triangle over the whole target
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x), dims.width - 1u);
    let y = min(u32(pos.y), dims.height - 1u);
    let k = (y * dims.width + x) * 3u;
    var c = clamp(vec3<f32>(image[k], image[k + 1u], image[k + 2u]), vec3<f32>(0.0), vec3<f32>(1.0));
    if (dims.srgb_target == 1u) {
        // an sRGB target encodes what it is given: hand it the decoded values
        c = pow(c, vec3<f32>(2.2));
    }
    return vec4<f32>(c, 1.0);
}
"#;

/// Draws the renderer's output buffer (linear-layout RGB `f32`) onto a colour
/// target of the same size: a full-screen triangle whose fragment shader reads
/// one pixel each.
pub struct Present {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    dims: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    srgb_target: bool,
}

impl Present {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_WGSL.into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(1, wgpu::BufferBindingType::Uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let dims = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            dims,
            bind_group: None,
            srgb_target: format.is_srgb(),
        }
    }

    /// Point the pass at an image buffer of `width` x `height` pixels.
    pub fn bind(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let dims = [width, height, u32::from(self.srgb_target), 0];
        queue.write_buffer(&self.dims, 0, bytemuck::cast_slice(&dims));
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dims.as_entire_binding(),
                },
            ],
        }));
    }

    /// Encode the pass onto `target` (nothing before [`Self::bind`]).
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use visloc_gsplat_core::splat::scene_from_splat;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    fn start() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console, js_name = error)]
        fn console_error(message: &str);
    }

    fn err(message: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&message.to_string())
    }

    /// A canvas that shows a gaussian scene: [`SplatViewer::create`] on the
    /// canvas, [`SplatViewer::load_splat`] with the scene, then
    /// [`SplatViewer::render`] once per frame the camera changes, awaiting
    /// each before the next call.
    #[wasm_bindgen]
    pub struct SplatViewer {
        inner: Rc<RefCell<Inner>>,
    }

    struct Inner {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        present: Present,
        /// The device while no scene is loaded (the renderer owns it after).
        ctx: Option<GpuContext>,
        /// Out of the cell while [`SplatViewer::render`] awaits it.
        renderer: Option<Renderer>,
        scene: Option<Scene>,
        /// A render is in flight: scene loads and resizes wait (`busy`).
        rendering: bool,
        /// The gaussians shown, and the adapter's name and backend.
        shown: u32,
        adapter: String,
    }

    impl Inner {
        fn context(&self) -> &GpuContext {
            match (&self.renderer, &self.ctx) {
                (Some(r), _) => r.context(),
                (None, Some(ctx)) => ctx,
                (None, None) => unreachable!("the viewer always holds its device"),
            }
        }

        /// Rebuild the renderer for the scene at the surface's size.
        fn rebuild(&mut self) -> Result<(), JsValue> {
            let ctx = match self.renderer.take() {
                Some(r) => r.into_context(),
                None => self.ctx.take().expect("the viewer holds its device"),
            };
            let Some(scene) = &self.scene else {
                self.ctx = Some(ctx);
                return Ok(());
            };
            let renderer =
                Renderer::new(ctx, scene, self.config.width, self.config.height).map_err(err)?;
            let c = renderer.context();
            self.present.bind(
                &c.device,
                &c.queue,
                renderer.output(),
                self.config.width,
                self.config.height,
            );
            self.shown = renderer.num_gaussians() as u32;
            self.renderer = Some(renderer);
            Ok(())
        }

        fn present_frame(&mut self) {
            let frame = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(&self.context().device, &self.config);
                    return;
                }
                _ => return,
            };
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let ctx = self.context();
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("present"),
                });
            self.present.draw(&mut encoder, &view);
            ctx.queue.submit(Some(encoder.finish()));
            ctx.queue.present(frame);
        }
    }

    #[wasm_bindgen]
    impl SplatViewer {
        /// Set up WebGPU on `canvas` (its current pixel size).
        pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<SplatViewer, JsValue> {
            let (width, height) = (canvas.width().max(1), canvas.height().max(1));
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .map_err(err)?;
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|e| err(format!("no WebGPU adapter: {e}")))?;
            let ctx = GpuContext::from_adapter(instance, &adapter)
                .await
                .map_err(err)?;
            ctx.device
                .on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
                    console_error(&format!("WebGPU: {e}"));
                }));
            let mut config = surface
                .get_default_config(&adapter, width, height)
                .ok_or_else(|| err("the canvas cannot be drawn on with this adapter"))?;
            // Splat colours are display values: an 8-bit target without the
            // sRGB encode shows them as they are.
            let caps = surface.get_capabilities(&adapter);
            if let Some(f) = caps.formats.iter().find(|f| !f.is_srgb()) {
                config.format = *f;
            }
            surface.configure(&ctx.device, &config);
            let present = Present::new(&ctx.device, config.format);
            let adapter = format!("{} ({:?})", ctx.adapter_info.name, ctx.adapter_info.backend);
            Ok(SplatViewer {
                inner: Rc::new(RefCell::new(Inner {
                    surface,
                    config,
                    present,
                    ctx: Some(ctx),
                    renderer: None,
                    scene: None,
                    rendering: false,
                    shown: 0,
                    adapter,
                })),
            })
        }

        /// The GPU adapter's name and backend.
        pub fn adapter(&self) -> String {
            self.inner.borrow().adapter.clone()
        }

        /// Show the scene of a `.splat` file; returns its centre (x, y, z) and
        /// size (see [`scene_bounds`]).
        pub fn load_splat(&self, bytes: &[u8]) -> Result<Vec<f32>, JsValue> {
            let scene = scene_from_splat(bytes).map_err(err)?;
            let (c, r) = scene_bounds(&scene);
            let mut inner = self.idle()?;
            inner.scene = Some(scene);
            inner.rebuild()?;
            Ok(vec![c[0], c[1], c[2], r])
        }

        /// The number of gaussians shown.
        pub fn num_gaussians(&self) -> u32 {
            self.inner.borrow().shown
        }

        /// Follow the canvas's new pixel size.
        pub fn resize(&self, width: u32, height: u32) -> Result<(), JsValue> {
            let mut inner = self.idle()?;
            let (width, height) = (width.max(1), height.max(1));
            if (width, height) == (inner.config.width, inner.config.height) {
                return Ok(());
            }
            inner.config.width = width;
            inner.config.height = height;
            let device = inner.context().device.clone();
            inner.surface.configure(&device, &inner.config);
            inner.rebuild()
        }

        /// Render the view from `eye` towards `target` (`up` the world's up,
        /// `fov_y` the vertical field of view in radians) and show it.
        pub fn render(
            &self,
            eye: Vec<f32>,
            target: Vec<f32>,
            up: Vec<f32>,
            fov_y: f32,
        ) -> js_sys::Promise {
            let inner = Rc::clone(&self.inner);
            wasm_bindgen_futures::future_to_promise(async move {
                // The renderer leaves the cell while the GPU works, so no
                // borrow is held across the await.
                let (mut renderer, w, h) = {
                    let mut state = inner.try_borrow_mut().map_err(|_| err("busy"))?;
                    let Some(renderer) = state.renderer.take() else {
                        return Ok(JsValue::UNDEFINED);
                    };
                    state.rendering = true;
                    (renderer, state.config.width, state.config.height)
                };
                let v = |s: &[f32]| Vector3::new(s[0], s[1], s[2]);
                let view = look_at(v(&eye), v(&target), v(&up), fov_y, w, h);
                renderer.render_async(&view, BACKGROUND).await;
                let mut state = inner.borrow_mut();
                state.renderer = Some(renderer);
                state.rendering = false;
                state.present_frame();
                Ok(JsValue::UNDEFINED)
            })
        }

        /// The state, unless a render is in flight.
        fn idle(&self) -> Result<std::cell::RefMut<'_, Inner>, JsValue> {
            let inner = self.inner.try_borrow_mut().map_err(|_| err("busy"))?;
            if inner.rendering {
                return Err(err("busy"));
            }
            Ok(inner)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::SplatViewer;

#[cfg(test)]
mod tests {
    use super::*;
    use visloc_gsplat_core::gaussian::Gaussian;

    #[test]
    fn look_at_puts_the_target_ahead_and_up_at_the_top() {
        let view = look_at(
            Vector3::new(0.0, 0.0, 5.0),
            Vector3::zeros(),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
            64,
            48,
        );
        let ahead = view.world_to_camera_point(Vector3::zeros());
        assert!(
            (ahead - Vector3::new(0.0, 0.0, 5.0)).norm() < 1e-6,
            "{ahead}"
        );
        let above = view.world_to_camera_point(Vector3::new(0.0, 1.0, 0.0));
        assert!(
            above.y < 0.0,
            "up is towards the image's top (y down): {above}"
        );
        let right = view.world_to_camera_point(Vector3::new(1.0, 0.0, 0.0));
        assert!(right.x > 0.0, "{right}");
        let (u, v, _) = view.camera.project(ahead);
        assert!((u - 32.0).abs() < 1e-4 && (v - 24.0).abs() < 1e-4);
    }

    #[test]
    fn scene_bounds_ignores_far_outliers() {
        let at = |x: f32| Gaussian {
            mean: Vector3::new(x, 0.0, 0.0),
            scale_log: Vector3::zeros(),
            rotation: nalgebra::Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: 0.0,
            sh_dc: [0.0; 3],
            sh_rest: Vec::new(),
            sh_degree: 0,
        };
        let mut gaussians: Vec<Gaussian> = (0..100).map(|i| at(i as f32 * 0.01)).collect();
        gaussians.push(at(1000.0));
        let (c, r) = scene_bounds(&Scene::new(gaussians, 0));
        assert!((c[0] - 0.5).abs() < 0.02, "{c:?}");
        assert!(r < 1.0, "{r}");
    }

    #[test]
    fn present_pass_draws_the_image_buffer() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("skipping present_pass_draws_the_image_buffer: no GPU adapter");
            return;
        };
        let (w, h) = (4u32, 2u32);
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pixels: Vec<f32> = (0..w * h)
            .flat_map(|i| [i as f32 / 8.0, 0.5, 1.0])
            .collect();
        let image = wgpu::util::DeviceExt::create_buffer_init(
            &ctx.device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&pixels),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut present = Present::new(&ctx.device, format);
        present.bind(&ctx.device, &ctx.queue, &image, w, h);
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        present.draw(&mut encoder, &target.create_view(&Default::default()));
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped")
            .to_vec();
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let px = &data[(y * 256 + x * 4) as usize..][..4];
                let want = [(i as f32 / 8.0 * 255.0).round() as u8, 128, 255, 255];
                for c in 0..4 {
                    assert!(
                        px[c].abs_diff(want[c]) <= 1,
                        "pixel ({x}, {y}): {px:?} vs {want:?}"
                    );
                }
            }
        }
    }
}
