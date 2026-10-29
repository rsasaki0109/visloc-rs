//! Runtime-loaded native CUDA 3D Gaussian Splatting rasterizer (opt-in).
//!
//! The shared library is built explicitly with `scripts/build_gsplat_cuda.sh`
//! from `native/gsplat_cuda/gsplat_cuda.cu`; Cargo never compiles CUDA, so
//! ordinary builds and docs.rs need no toolkit. The kernels mirror the wgpu
//! renderer's maths and are checked against the CPU f64 oracle
//! (`visloc_gsplat_core::backward`) by this crate's tests, which run when
//! `VISLOC_GSPLAT_CUDA_LIB` points at the built library.
//!
//! Gaussians use the [`PackedScene`] layout, and gradients come back in the
//! same layout: 10 transform floats (mean, un-normalised quaternion `wxyz`,
//! log-scale), the opacity logit and the SH block per gaussian.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_float, c_int, c_void, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use libloading::Library;
use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_render::packing::PackedScene;

/// Environment variable naming the built shared library (used by the tests
/// and examples).
pub const LIBRARY_ENV: &str = "VISLOC_GSPLAT_CUDA_LIB";

/// Floats per gaussian in the screen-space gradient record: `du dv dA dB dC
/// dopacity dr dg db` and the refine weight (sum over pixels of
/// `|dL/du| W + |dL/dv| H`).
pub const SCREEN_GRAD_FLOATS: usize = 10;

/// Camera as the native ABI takes it (`GscCamera` in the CUDA source).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GscCamera {
    /// World-to-camera rotation, row-major.
    pub rot: [f32; 9],
    /// World-to-camera translation.
    pub t: [f32; 3],
    /// Camera centre in world coordinates.
    pub center: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl GscCamera {
    pub fn from_view(view: &CameraView) -> Self {
        let r = &view.rotation;
        let c = view.camera_center();
        Self {
            rot: [
                r[(0, 0)],
                r[(0, 1)],
                r[(0, 2)],
                r[(1, 0)],
                r[(1, 1)],
                r[(1, 2)],
                r[(2, 0)],
                r[(2, 1)],
                r[(2, 2)],
            ],
            t: [view.translation.x, view.translation.y, view.translation.z],
            center: [c.x, c.y, c.z],
            fx: view.camera.fx,
            fy: view.camera.fy,
            cx: view.camera.cx,
            cy: view.camera.cy,
            width: view.camera.width,
            height: view.camera.height,
        }
    }
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type LastErrorFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type StageNamesFn = unsafe extern "C" fn() -> *const c_char;
type SetSceneFn = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    *const c_float,
    *const c_float,
    *const c_float,
) -> c_int;
type RenderFn = unsafe extern "C" fn(
    *mut c_void,
    *const GscCamera,
    *const c_float,
    c_int,
    *mut c_float,
    *mut u32,
) -> c_int;
type BackwardFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_float,
    *mut c_float,
    *mut c_float,
    *mut c_float,
    *mut c_float,
) -> c_int;
type StageTimesFn = unsafe extern "C" fn(*mut c_void, *mut c_float, c_int) -> c_int;

#[derive(Debug)]
pub enum GsplatCudaError {
    Load { path: PathBuf, message: String },
    AbiVersion(u32),
    NullContext,
    Input(String),
    Runtime(String),
}

impl fmt::Display for GsplatCudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, message } => {
                write!(
                    f,
                    "load native CUDA rasterizer {}: {message}",
                    path.display()
                )
            }
            Self::AbiVersion(version) => write!(
                f,
                "native CUDA rasterizer ABI {version}, expected {}",
                CudaRasterizer::ABI_VERSION
            ),
            Self::NullContext => write!(f, "native CUDA rasterizer returned a null context"),
            Self::Input(message) => write!(f, "native CUDA rasterizer input: {message}"),
            Self::Runtime(message) => write!(f, "native CUDA rasterizer failed: {message}"),
        }
    }
}

impl std::error::Error for GsplatCudaError {}

/// One rendered frame.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Row-major RGB, `width * height * 3` floats (background included).
    pub rgb: Vec<f32>,
    /// Number of (tile, gaussian) pairs rasterized.
    pub num_intersections: u32,
}

/// Gradients of `L = sum_px dot(d_image[px], C[px])` for the last frame, in
/// the [`PackedScene`] layouts. Gaussians that were not visible get zeros.
#[derive(Debug, Clone)]
pub struct Gradients {
    pub transforms: Vec<f32>,
    pub opacity: Vec<f32>,
    pub sh: Vec<f32>,
    /// [`SCREEN_GRAD_FLOATS`] per gaussian.
    pub screen: Vec<f32>,
}

pub struct CudaRasterizer {
    _library: Library,
    context: NonNull<c_void>,
    destroy: DestroyFn,
    last_error: LastErrorFn,
    stage_names: Vec<String>,
    set_scene: SetSceneFn,
    render: RenderFn,
    backward: BackwardFn,
    stage_times: StageTimesFn,
    num_gaussians: usize,
    sh_floats: usize,
    pixels: usize,
}

impl fmt::Debug for CudaRasterizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaRasterizer")
            .field("num_gaussians", &self.num_gaussians)
            .finish_non_exhaustive()
    }
}

impl CudaRasterizer {
    pub const ABI_VERSION: u32 = 1;

    /// Load the library named by [`LIBRARY_ENV`], or `None` when it is unset.
    pub fn from_env() -> Option<Result<Self, GsplatCudaError>> {
        std::env::var_os(LIBRARY_ENV).map(Self::load)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, GsplatCudaError> {
        let path = path.as_ref();
        let load_error = |message: String| GsplatCudaError::Load {
            path: path.to_path_buf(),
            message,
        };
        // SAFETY: loading runs the library's initialisers; the library is
        // ours (native/gsplat_cuda) and its ABI version is checked below
        // before any other entry point is called.
        let library = unsafe { Library::new(path) }.map_err(|e| load_error(e.to_string()))?;
        // SAFETY: every symbol is requested under the exact name and
        // signature declared in native/gsplat_cuda/gsplat_cuda.cu, and the
        // copied function pointers are only called while `library` is owned
        // by the struct (`_library`), so the code stays mapped.
        let (
            abi_version,
            create,
            destroy,
            last_error,
            stage_names,
            set_scene,
            render,
            backward,
            stage_times,
        ): (
            AbiVersionFn,
            CreateFn,
            DestroyFn,
            LastErrorFn,
            StageNamesFn,
            SetSceneFn,
            RenderFn,
            BackwardFn,
            StageTimesFn,
        ) = unsafe {
            (
                *library
                    .get(b"visloc_gsplat_cuda_abi_version\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_create\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_destroy\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_last_error\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_stage_names\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_set_scene\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_render\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_backward\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_gsplat_cuda_stage_times\0")
                    .map_err(|e| load_error(e.to_string()))?,
            )
        };
        // SAFETY: takes no arguments and only returns a constant.
        let version = unsafe { abi_version() };
        if version != Self::ABI_VERSION {
            return Err(GsplatCudaError::AbiVersion(version));
        }
        // SAFETY: returns a pointer to a static NUL-terminated string.
        let stage_names = unsafe { CStr::from_ptr(stage_names()) }
            .to_string_lossy()
            .split(',')
            .map(str::to_owned)
            .collect();
        // SAFETY: a null return is rejected; the non-null context is owned by
        // `self` and released exactly once through `destroy` in Drop.
        let context = NonNull::new(unsafe { create() }).ok_or(GsplatCudaError::NullContext)?;
        let rasterizer = Self {
            _library: library,
            context,
            destroy,
            last_error,
            stage_names,
            set_scene,
            render,
            backward,
            stage_times,
            num_gaussians: 0,
            sh_floats: 0,
            pixels: 0,
        };
        // create() records a device failure (e.g. no CUDA device) in the
        // context's error string instead of returning null.
        let message = rasterizer.error_message();
        if !message.is_empty() {
            return Err(GsplatCudaError::Runtime(message));
        }
        Ok(rasterizer)
    }

    fn error_message(&self) -> String {
        // SAFETY: the context is live; the returned string is owned by it and
        // copied before any other call can change it.
        unsafe { CStr::from_ptr((self.last_error)(self.context.as_ptr())) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(&self, code: c_int) -> Result<(), GsplatCudaError> {
        if code == 0 {
            Ok(())
        } else {
            Err(GsplatCudaError::Runtime(self.error_message()))
        }
    }

    /// Upload a packed scene (replaces the previous one).
    pub fn set_scene(&mut self, scene: &PackedScene) -> Result<(), GsplatCudaError> {
        let n = scene.num_gaussians;
        let cpc = scene.sh_coeffs_per_channel;
        if scene.transforms.len() != n * 10
            || scene.opacity.len() != n
            || scene.sh.len() != n * 3 * cpc
        {
            return Err(GsplatCudaError::Input("packed scene buffer sizes".into()));
        }
        let n_c =
            c_int::try_from(n).map_err(|_| GsplatCudaError::Input("too many gaussians".into()))?;
        // SAFETY: the buffers hold exactly the sizes checked above, which are
        // what the native side reads for `n` gaussians of this SH degree.
        let code = unsafe {
            (self.set_scene)(
                self.context.as_ptr(),
                n_c,
                scene.sh_degree as c_int,
                scene.transforms.as_ptr(),
                scene.opacity.as_ptr(),
                scene.sh.as_ptr(),
            )
        };
        self.check(code)?;
        self.num_gaussians = n;
        self.sh_floats = n * 3 * cpc;
        Ok(())
    }

    /// Render the uploaded scene from `view` with SH evaluated up to
    /// `sh_active_degree` (at most the scene's degree).
    pub fn render(
        &mut self,
        view: &CameraView,
        bg: [f32; 3],
        sh_active_degree: u32,
    ) -> Result<Frame, GsplatCudaError> {
        let cam = GscCamera::from_view(view);
        let pixels = cam.width as usize * cam.height as usize;
        let mut rgb = vec![0.0f32; pixels * 3];
        let mut num_intersections = 0u32;
        // SAFETY: `rgb` holds width * height * 3 floats, the size the native
        // side writes; `cam`, `bg` and the counter outlive the call.
        let code = unsafe {
            (self.render)(
                self.context.as_ptr(),
                &cam,
                bg.as_ptr(),
                sh_active_degree as c_int,
                rgb.as_mut_ptr(),
                &mut num_intersections,
            )
        };
        self.check(code)?;
        self.pixels = pixels;
        Ok(Frame {
            rgb,
            num_intersections,
        })
    }

    /// Gradients for the last [`render`](Self::render) given `d_image`
    /// (row-major RGB, `width * height * 3`).
    pub fn backward(&mut self, d_image: &[f32]) -> Result<Gradients, GsplatCudaError> {
        if d_image.len() != self.pixels * 3 {
            return Err(GsplatCudaError::Input(format!(
                "d_image has {} floats, the last frame needs {}",
                d_image.len(),
                self.pixels * 3
            )));
        }
        let n = self.num_gaussians;
        let mut grads = Gradients {
            transforms: vec![0.0; n * 10],
            opacity: vec![0.0; n],
            sh: vec![0.0; self.sh_floats],
            screen: vec![0.0; n * SCREEN_GRAD_FLOATS],
        };
        // SAFETY: every output holds the size the native side writes for the
        // uploaded scene, and `d_image` the last frame's pixel count.
        let code = unsafe {
            (self.backward)(
                self.context.as_ptr(),
                d_image.as_ptr(),
                grads.transforms.as_mut_ptr(),
                grads.opacity.as_mut_ptr(),
                grads.sh.as_mut_ptr(),
                grads.screen.as_mut_ptr(),
            )
        };
        self.check(code)?;
        Ok(grads)
    }

    /// GPU milliseconds of each stage of the last render and backward
    /// (between CUDA events), by stage name.
    pub fn stage_times(&self) -> Vec<(String, f32)> {
        let mut ms = vec![0.0f32; self.stage_names.len()];
        // SAFETY: `ms` holds `stage_names.len()` floats and its length is
        // passed as the capacity.
        let count = unsafe {
            (self.stage_times)(self.context.as_ptr(), ms.as_mut_ptr(), ms.len() as c_int)
        };
        let count = usize::try_from(count).unwrap_or(0);
        self.stage_names
            .iter()
            .cloned()
            .zip(ms)
            .take(count)
            .collect()
    }
}

impl Drop for CudaRasterizer {
    fn drop(&mut self) {
        // SAFETY: the context came from `create` and is destroyed exactly once.
        unsafe { (self.destroy)(self.context.as_ptr()) };
    }
}
