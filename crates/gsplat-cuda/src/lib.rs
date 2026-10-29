//! Runtime-loaded native CUDA 3D Gaussian Splatting rasterizer and training
//! step (opt-in).
//!
//! The shared library is built explicitly with `scripts/build_gsplat_cuda.sh`
//! from `native/gsplat_cuda/gsplat_cuda.cu`; Cargo never compiles CUDA, so
//! ordinary builds and docs.rs need no toolkit. The kernels mirror the wgpu
//! renderer and trainer, and are checked against the CPU f64 oracle
//! (`visloc_gsplat_core::backward`) by this crate's tests, which run when
//! `VISLOC_GSPLAT_CUDA_LIB` points at the built library.
//!
//! Gaussians use the [`PackedScene`] layout, and gradients come back in the
//! same layout: 10 transform floats (mean, un-normalised quaternion `wxyz`,
//! log-scale), the opacity logit and the SH block per gaussian.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_double, c_float, c_int, c_void, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

pub mod trainer;

use libloading::Library;
use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_render::packing::{PackedScene, TRANSFORM_FLOATS};

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

/// One optimisation step (`GscStepParams` in the CUDA source): the wgpu
/// trainer's step on uploaded image `image`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepParams {
    pub camera: GscCamera,
    /// Index into the images given to [`CudaGsplat::set_images`].
    pub image: u32,
    pub sh_active_degree: i32,
    pub background: [f32; 3],
    /// Loss = (1 - w) L1 + w (1 - SSIM).
    pub ssim_weight: f32,
    /// Learning rates: mean, quaternion, log-scale, opacity logit, SH DC,
    /// SH rest.
    pub lr: [f32; 6],
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Adam bias corrections `1 - beta1^t`, `1 - beta2^t`.
    pub bc1: f32,
    pub bc2: f32,
    /// brush's auxiliary losses on log-scales and opacity logits (0 = off).
    pub scale_aux_coef: f32,
    pub opacity_aux_coef: f32,
    /// Accumulate brush's refine statistics (0 = no).
    pub accumulate_stats: u32,
    pub noise_seed: u32,
    /// `lr_mean * mean_noise_weight` (0 = no mean noise).
    pub noise_scale: f32,
    pub noise_max: f32,
    /// Run the SH Adam only on the gaussians blended this frame (the
    /// others keep their SH and its moments; 0 = dense, as the wgpu trainer).
    pub sparse_sh_adam: u32,
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
type NumGaussiansFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type GetSceneFn =
    unsafe extern "C" fn(*mut c_void, *mut c_float, *mut c_float, *mut c_float) -> c_int;
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
type SetImagesFn = unsafe extern "C" fn(*mut c_void, c_int, u32, u32, *const u32) -> c_int;
type TrainStepFn = unsafe extern "C" fn(*mut c_void, *const StepParams) -> c_int;
type LossGradFn = unsafe extern "C" fn(*mut c_void, u32, c_float, *mut c_float) -> c_int;
type TakeLossFn =
    unsafe extern "C" fn(*mut c_void, *mut c_double, *mut c_double, *mut u32) -> c_int;
type RefineStatsFn = unsafe extern "C" fn(*mut c_void, *mut c_float, *mut c_float) -> c_int;
type ApplyActionsFn = unsafe extern "C" fn(*mut c_void, *const u32, u32, *mut c_int) -> c_int;
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
                write!(f, "load native CUDA gsplat {}: {message}", path.display())
            }
            Self::AbiVersion(version) => write!(
                f,
                "native CUDA gsplat ABI {version}, expected {}",
                CudaGsplat::ABI_VERSION
            ),
            Self::NullContext => write!(f, "native CUDA gsplat returned a null context"),
            Self::Input(message) => write!(f, "native CUDA gsplat input: {message}"),
            Self::Runtime(message) => write!(f, "native CUDA gsplat failed: {message}"),
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

/// Loss sums since the last [`CudaGsplat::take_loss`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossSums {
    /// Sum over steps of the mean L1.
    pub l1: f64,
    /// Sum over steps of the SSIM summed over pixels and channels (0 when
    /// the SSIM term is off).
    pub ssim: f64,
    pub steps: u32,
}

/// brush's refine statistics per gaussian since the last refine.
#[derive(Debug, Clone)]
pub struct RefineStats {
    /// Max refine weight over the frames that blended the gaussian.
    pub weight_max: Vec<f32>,
    /// Number of such frames.
    pub seen: Vec<f32>,
}

struct Api {
    destroy: DestroyFn,
    last_error: LastErrorFn,
    set_scene: SetSceneFn,
    num_gaussians: NumGaussiansFn,
    get_scene: GetSceneFn,
    render: RenderFn,
    backward: BackwardFn,
    set_images: SetImagesFn,
    train_step: TrainStepFn,
    loss_grad: LossGradFn,
    take_loss: TakeLossFn,
    refine_stats: RefineStatsFn,
    apply_actions: ApplyActionsFn,
    stage_times: StageTimesFn,
}

/// The native context: a scene, its Adam state and the training images on
/// the device.
pub struct CudaGsplat {
    api: Api,
    context: NonNull<c_void>,
    stage_names: Vec<String>,
    sh_degree: u32,
    pixels: usize,
    // Declared last: dropped after `context` is destroyed in Drop.
    _library: Library,
}

impl fmt::Debug for CudaGsplat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaGsplat")
            .field("sh_degree", &self.sh_degree)
            .finish_non_exhaustive()
    }
}

/// Resolve `name` from `library` as a function pointer of type `T`.
///
/// # Safety
///
/// `T` must be the exact signature the library declares for `name`, and the
/// pointer must not outlive `library`.
unsafe fn symbol<T: Copy>(
    library: &Library,
    name: &str,
    path: &Path,
) -> Result<T, GsplatCudaError> {
    let c_name = format!("{name}\0");
    // SAFETY: forwarded to the caller's contract.
    unsafe { library.get::<T>(c_name.as_bytes()) }
        .map(|s| *s)
        .map_err(|e| GsplatCudaError::Load {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
}

impl CudaGsplat {
    pub const ABI_VERSION: u32 = 3;

    /// Load the library named by [`LIBRARY_ENV`], or `None` when it is unset.
    pub fn from_env() -> Option<Result<Self, GsplatCudaError>> {
        std::env::var_os(LIBRARY_ENV).map(Self::load)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, GsplatCudaError> {
        let path = path.as_ref();
        // SAFETY: loading runs the library's initialisers; the library is
        // ours (native/gsplat_cuda) and its ABI version is checked below
        // before any other entry point is called.
        let library = unsafe { Library::new(path) }.map_err(|e| GsplatCudaError::Load {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        // SAFETY: each symbol's type is the signature declared in
        // native/gsplat_cuda/gsplat_cuda.cu, and the pointers live in `Self`
        // next to the `Library` that keeps them mapped.
        let (abi_version, create, stage_names, api) = unsafe {
            let abi_version: AbiVersionFn =
                symbol(&library, "visloc_gsplat_cuda_abi_version", path)?;
            // Check the version before resolving symbols that may not exist
            // in an older library.
            let version = abi_version();
            if version != Self::ABI_VERSION {
                return Err(GsplatCudaError::AbiVersion(version));
            }
            let create: CreateFn = symbol(&library, "visloc_gsplat_cuda_create", path)?;
            let stage_names: StageNamesFn =
                symbol(&library, "visloc_gsplat_cuda_stage_names", path)?;
            let api = Api {
                destroy: symbol(&library, "visloc_gsplat_cuda_destroy", path)?,
                last_error: symbol(&library, "visloc_gsplat_cuda_last_error", path)?,
                set_scene: symbol(&library, "visloc_gsplat_cuda_set_scene", path)?,
                num_gaussians: symbol(&library, "visloc_gsplat_cuda_num_gaussians", path)?,
                get_scene: symbol(&library, "visloc_gsplat_cuda_get_scene", path)?,
                render: symbol(&library, "visloc_gsplat_cuda_render", path)?,
                backward: symbol(&library, "visloc_gsplat_cuda_backward", path)?,
                set_images: symbol(&library, "visloc_gsplat_cuda_set_images", path)?,
                train_step: symbol(&library, "visloc_gsplat_cuda_train_step", path)?,
                loss_grad: symbol(&library, "visloc_gsplat_cuda_loss_grad", path)?,
                take_loss: symbol(&library, "visloc_gsplat_cuda_take_loss", path)?,
                refine_stats: symbol(&library, "visloc_gsplat_cuda_get_refine_stats", path)?,
                apply_actions: symbol(&library, "visloc_gsplat_cuda_apply_actions", path)?,
                stage_times: symbol(&library, "visloc_gsplat_cuda_stage_times", path)?,
            };
            (version, create, stage_names, api)
        };
        debug_assert_eq!(abi_version, Self::ABI_VERSION);
        // SAFETY: returns a pointer to a static NUL-terminated string.
        let stage_names = unsafe { CStr::from_ptr(stage_names()) }
            .to_string_lossy()
            .split(',')
            .map(str::to_owned)
            .collect();
        // SAFETY: a null return is rejected; the non-null context is owned by
        // `self` and released exactly once through `destroy` in Drop.
        let context = NonNull::new(unsafe { create() }).ok_or(GsplatCudaError::NullContext)?;
        let gs = Self {
            api,
            context,
            stage_names,
            sh_degree: 0,
            pixels: 0,
            _library: library,
        };
        // create() records a device failure (e.g. no CUDA device) in the
        // context's error string instead of returning null.
        let message = gs.error_message();
        if !message.is_empty() {
            return Err(GsplatCudaError::Runtime(message));
        }
        Ok(gs)
    }

    fn ctx(&self) -> *mut c_void {
        self.context.as_ptr()
    }

    fn error_message(&self) -> String {
        // SAFETY: the context is live; the returned string is owned by it and
        // copied before any other call can change it.
        unsafe { CStr::from_ptr((self.api.last_error)(self.ctx())) }
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

    fn sh_floats_per_gaussian(&self) -> usize {
        3 * ((self.sh_degree + 1) * (self.sh_degree + 1)) as usize
    }

    /// Upload a packed scene, replacing the current one; the Adam moments
    /// and refine statistics start at zero.
    pub fn set_scene(&mut self, scene: &PackedScene) -> Result<(), GsplatCudaError> {
        let n = scene.num_gaussians;
        let cpc = scene.sh_coeffs_per_channel;
        if scene.transforms.len() != n * TRANSFORM_FLOATS
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
            (self.api.set_scene)(
                self.ctx(),
                n_c,
                scene.sh_degree as c_int,
                scene.transforms.as_ptr(),
                scene.opacity.as_ptr(),
                scene.sh.as_ptr(),
            )
        };
        self.check(code)?;
        self.sh_degree = scene.sh_degree;
        Ok(())
    }

    pub fn num_gaussians(&self) -> usize {
        // SAFETY: the context is live.
        let n = unsafe { (self.api.num_gaussians)(self.ctx()) };
        usize::try_from(n).unwrap_or(0)
    }

    /// Download the current scene.
    pub fn scene(&self) -> Result<PackedScene, GsplatCudaError> {
        let n = self.num_gaussians();
        let mut out = PackedScene {
            transforms: vec![0.0; n * TRANSFORM_FLOATS],
            sh: vec![0.0; n * self.sh_floats_per_gaussian()],
            sh_coeffs_per_channel: self.sh_floats_per_gaussian() / 3,
            opacity: vec![0.0; n],
            num_gaussians: n,
            sh_degree: self.sh_degree,
        };
        // SAFETY: every output holds the size the native side writes for the
        // current gaussian count and SH degree.
        let code = unsafe {
            (self.api.get_scene)(
                self.ctx(),
                out.transforms.as_mut_ptr(),
                out.opacity.as_mut_ptr(),
                out.sh.as_mut_ptr(),
            )
        };
        self.check(code)?;
        Ok(out)
    }

    /// Render the current scene from `view` with SH evaluated up to
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
            (self.api.render)(
                self.ctx(),
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
        let n = self.num_gaussians();
        let mut grads = Gradients {
            transforms: vec![0.0; n * TRANSFORM_FLOATS],
            opacity: vec![0.0; n],
            sh: vec![0.0; n * self.sh_floats_per_gaussian()],
            screen: vec![0.0; n * SCREEN_GRAD_FLOATS],
        };
        // SAFETY: every output holds the size the native side writes for the
        // current scene, and `d_image` the last frame's pixel count.
        let code = unsafe {
            (self.api.backward)(
                self.ctx(),
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

    /// Upload the training images: `count` images of `width x height`, one
    /// packed RGBA8 `u32` per pixel (R in the low byte; alpha below 128
    /// leaves the pixel out of the loss), image-major.
    pub fn set_images(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u32],
    ) -> Result<(), GsplatCudaError> {
        let per = width as usize * height as usize;
        if per == 0 || !rgba.len().is_multiple_of(per) {
            return Err(GsplatCudaError::Input(
                "image buffer is not a whole number of images".into(),
            ));
        }
        let count = c_int::try_from(rgba.len() / per)
            .map_err(|_| GsplatCudaError::Input("too many images".into()))?;
        // SAFETY: `rgba` holds `count` images of width * height pixels.
        let code =
            unsafe { (self.api.set_images)(self.ctx(), count, width, height, rgba.as_ptr()) };
        self.check(code)
    }

    /// One optimisation step (see [`StepParams`]).
    pub fn train_step(&mut self, params: &StepParams) -> Result<(), GsplatCudaError> {
        // SAFETY: `params` is a live `repr(C)` struct of the declared layout.
        let code = unsafe { (self.api.train_step)(self.ctx(), params) };
        self.check(code)?;
        self.pixels = params.camera.width as usize * params.camera.height as usize;
        Ok(())
    }

    /// The training loss's image gradient for the last render against
    /// uploaded image `image` (what a step computes). The loss sums count
    /// towards [`take_loss`](Self::take_loss).
    pub fn loss_grad(&mut self, image: u32, ssim_weight: f32) -> Result<Vec<f32>, GsplatCudaError> {
        let mut d_image = vec![0.0f32; self.pixels * 3];
        // SAFETY: `d_image` holds the last frame's width * height * 3 floats.
        let code =
            unsafe { (self.api.loss_grad)(self.ctx(), image, ssim_weight, d_image.as_mut_ptr()) };
        self.check(code)?;
        Ok(d_image)
    }

    /// Loss sums since the last call (resets them).
    pub fn take_loss(&mut self) -> Result<LossSums, GsplatCudaError> {
        let mut sums = LossSums {
            l1: 0.0,
            ssim: 0.0,
            steps: 0,
        };
        // SAFETY: the outputs are live scalars.
        let code = unsafe {
            (self.api.take_loss)(self.ctx(), &mut sums.l1, &mut sums.ssim, &mut sums.steps)
        };
        self.check(code)?;
        Ok(sums)
    }

    /// brush's refine statistics since the last refine.
    pub fn refine_stats(&self) -> Result<RefineStats, GsplatCudaError> {
        let n = self.num_gaussians();
        let mut stats = RefineStats {
            weight_max: vec![0.0; n],
            seen: vec![0.0; n],
        };
        // SAFETY: both outputs hold one float per current gaussian.
        let code = unsafe {
            (self.api.refine_stats)(
                self.ctx(),
                stats.weight_max.as_mut_ptr(),
                stats.seen.as_mut_ptr(),
            )
        };
        self.check(code)?;
        Ok(stats)
    }

    /// Apply one action per gaussian (0 prune, 1 keep, 2 clone, 4 brush
    /// split) to the scene and its Adam moments, zeroing the refine
    /// statistics. Returns the new gaussian count.
    pub fn apply_actions(&mut self, actions: &[u32], seed: u32) -> Result<usize, GsplatCudaError> {
        if actions.len() != self.num_gaussians() {
            return Err(GsplatCudaError::Input(format!(
                "{} actions for {} gaussians",
                actions.len(),
                self.num_gaussians()
            )));
        }
        let mut n_after: c_int = 0;
        // SAFETY: `actions` holds one entry per current gaussian.
        let code =
            unsafe { (self.api.apply_actions)(self.ctx(), actions.as_ptr(), seed, &mut n_after) };
        self.check(code)?;
        Ok(usize::try_from(n_after).unwrap_or(0))
    }

    /// GPU milliseconds of each stage of the last render, backward or step
    /// (between CUDA events; 0 for stages that did not run), by name.
    pub fn stage_times(&self) -> Vec<(String, f32)> {
        let mut ms = vec![0.0f32; self.stage_names.len()];
        // SAFETY: `ms` holds `stage_names.len()` floats and its length is
        // passed as the capacity.
        let count =
            unsafe { (self.api.stage_times)(self.ctx(), ms.as_mut_ptr(), ms.len() as c_int) };
        let count = usize::try_from(count).unwrap_or(0);
        self.stage_names
            .iter()
            .cloned()
            .zip(ms)
            .take(count)
            .collect()
    }
}

impl Drop for CudaGsplat {
    fn drop(&mut self) {
        // SAFETY: the context came from `create` and is destroyed exactly once,
        // before `_library` (declared last) is dropped.
        unsafe { (self.api.destroy)(self.ctx()) };
    }
}
