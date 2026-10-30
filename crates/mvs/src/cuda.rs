//! PatchMatch on CUDA: [`crate::gpu`]'s compute shaders as CUDA kernels
//! (`native/mvs_cuda/mvs_cuda.cu`), for machines where wgpu cannot reach the
//! GPU (WSL2). The kernels follow the shader line by line, so the CPU path's
//! sampling, view selection, refinement and random numbers; they sample the
//! sources in f32 where the shader's texture holds half floats.
//!
//! The shared library is built explicitly with `scripts/build_mvs_cuda.sh`
//! and loaded at runtime ([`LIBRARY_ENV`] names it for the tests and the
//! `mvs` example); Cargo never compiles CUDA.

// The native library's C API.
#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Mutex, PoisonError};

use libloading::Library;

use crate::deformable::{self, Class};
use crate::pass_setup::{Params, PassSetup, SourceCam};
use crate::patch_match::{
    finish_pass_depth_map, DepthMap, DetailRestorer, PatchMatchOptions, MAX_COST,
};
use crate::planar_prior::{PlanarPrior, PlanarPriorOptions};
use crate::workspace::Workspace;

/// Environment variable naming the built shared library.
pub const LIBRARY_ENV: &str = "VISLOC_MVS_CUDA_LIB";

/// The native library's ABI version (`kAbiVersion` in the CUDA source).
const ABI_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("load native CUDA MVS {}: {message}", path.display())]
    Load { path: PathBuf, message: String },
    #[error("native CUDA MVS ABI {0}, expected {ABI_VERSION}")]
    AbiVersion(u32),
    #[error("native CUDA MVS returned a null context")]
    NullContext,
    #[error("native CUDA MVS failed: {0}")]
    Runtime(String),
}

/// A pass's inputs as the native side takes them (`MvsInputs`).
#[repr(C)]
struct Inputs {
    width: u32,
    height: u32,
    num_sources: u32,
    texture_width: u32,
    texture_height: u32,
    sources: *const SourceCam,
    layers: *const f32,
    image: *const f32,
    prior_len: u32,
    priors: *const f32,
    init_planes: *const [f32; 4],
    prior_planes: *const [f32; 4],
    pre_costs: u32,
    classes: u32,
    deformable: u32,
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type LastErrorFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type BeginFn = unsafe extern "C" fn(*mut c_void, *const Inputs, *const Params) -> c_int;
type ClassifyFn = unsafe extern "C" fn(*mut c_void, *const Params, *mut u32) -> c_int;
type SetClassesFn = unsafe extern "C" fn(*mut c_void, *const u32) -> c_int;
type FindAnchorsFn = unsafe extern "C" fn(*mut c_void, *const Params, *const u32) -> c_int;
type PropagateFn = unsafe extern "C" fn(*mut c_void, *const Params) -> c_int;
type ReadFn = unsafe extern "C" fn(*mut c_void, *mut [f32; 4], *mut f32) -> c_int;

struct Api {
    destroy: DestroyFn,
    last_error: LastErrorFn,
    begin: BeginFn,
    classify: ClassifyFn,
    set_classes: SetClassesFn,
    find_anchors: FindAnchorsFn,
    propagate: PropagateFn,
    read: ReadFn,
}

/// The native context: a pass's device buffers.
struct Context(NonNull<c_void>);

// SAFETY: the context is host memory holding device pointers of the
// process's primary CUDA context, which the CUDA runtime lets any thread
// use; it is only reached through the mutex in `CudaPatchMatch`.
unsafe impl Send for Context {}

/// The CUDA PatchMatch: the loaded library and its context. One pass runs
/// at a time.
pub struct CudaPatchMatch {
    api: Api,
    context: Mutex<Context>,
    // Declared last: dropped after `context` is destroyed in Drop.
    _library: Library,
}

/// A pass's planes, costs, the pixels sorted for the next pass (empty
/// unless asked for) and, in a deformable pass, the classes it ran with.
type PassOutput = (Vec<[f32; 4]>, Vec<f32>, Vec<Class>, Option<Vec<Class>>);

/// Resolves `name` from `library` as a function pointer of type `T`.
///
/// # Safety
///
/// `T` must be the exact signature the library declares for `name`, and the
/// pointer must not outlive `library`.
unsafe fn symbol<T: Copy>(library: &Library, name: &str, path: &Path) -> Result<T, CudaError> {
    let c_name = format!("{name}\0");
    // SAFETY: forwarded to the caller's contract.
    unsafe { library.get::<T>(c_name.as_bytes()) }
        .map(|s| *s)
        .map_err(|e| CudaError::Load {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
}

fn class_code(class: Class) -> u32 {
    match class {
        Class::Weak => 0,
        Class::Strong => 1,
        Class::Unknown => 2,
    }
}

impl CudaPatchMatch {
    /// Loads the library named by [`LIBRARY_ENV`], or `None` when it is
    /// unset.
    pub fn from_env() -> Option<Result<Self, CudaError>> {
        std::env::var_os(LIBRARY_ENV).map(Self::load)
    }

    /// Loads the library at `path` and creates a context on the first CUDA
    /// device.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let path = path.as_ref();
        // SAFETY: loading runs the library's initialisers; the library is
        // ours (native/mvs_cuda) and its ABI version is checked below before
        // any other entry point is called.
        let library = unsafe { Library::new(path) }.map_err(|e| CudaError::Load {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        // SAFETY: each symbol's type is the signature declared in
        // native/mvs_cuda/mvs_cuda.cu, and the pointers live in `Self` next
        // to the `Library` that keeps them mapped.
        let (create, api) = unsafe {
            let abi_version: AbiVersionFn = symbol(&library, "visloc_mvs_cuda_abi_version", path)?;
            let version = abi_version();
            if version != ABI_VERSION {
                return Err(CudaError::AbiVersion(version));
            }
            let create: CreateFn = symbol(&library, "visloc_mvs_cuda_create", path)?;
            let api = Api {
                destroy: symbol(&library, "visloc_mvs_cuda_destroy", path)?,
                last_error: symbol(&library, "visloc_mvs_cuda_last_error", path)?,
                begin: symbol(&library, "visloc_mvs_cuda_begin", path)?,
                classify: symbol(&library, "visloc_mvs_cuda_classify", path)?,
                set_classes: symbol(&library, "visloc_mvs_cuda_set_classes", path)?,
                find_anchors: symbol(&library, "visloc_mvs_cuda_find_anchors", path)?,
                propagate: symbol(&library, "visloc_mvs_cuda_propagate", path)?,
                read: symbol(&library, "visloc_mvs_cuda_read", path)?,
            };
            (create, api)
        };
        // SAFETY: a null return is rejected; the non-null context is owned
        // by `self` and released exactly once through `destroy` in Drop.
        let context = NonNull::new(unsafe { create() }).ok_or(CudaError::NullContext)?;
        let cuda = Self {
            api,
            context: Mutex::new(Context(context)),
            _library: library,
        };
        // create() records a device failure (no CUDA device) in the
        // context's error string instead of returning null.
        let message = cuda.error_message(context.as_ptr());
        if !message.is_empty() {
            return Err(CudaError::Runtime(message));
        }
        Ok(cuda)
    }

    fn error_message(&self, context: *mut c_void) -> String {
        // SAFETY: the context is live; the returned string is owned by it
        // and copied before any other call can change it.
        unsafe { CStr::from_ptr((self.api.last_error)(context)) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(&self, context: *mut c_void, code: c_int) -> Result<(), CudaError> {
        if code == 0 {
            Ok(())
        } else {
            Err(CudaError::Runtime(self.error_message(context)))
        }
    }

    /// [`crate::patch_match::estimate`] on CUDA.
    ///
    /// # Panics
    ///
    /// If a CUDA call fails (the device runs out of memory, say).
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

    /// [`crate::patch_match::estimate_from`] on CUDA.
    ///
    /// # Panics
    ///
    /// If a CUDA call fails.
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

    /// [`crate::patch_match::estimate_with_prior`] on CUDA.
    ///
    /// # Panics
    ///
    /// If a CUDA call fails.
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

    /// [`crate::patch_match::photometric_score`] on CUDA: the first pass
    /// alone, started from `map`.
    ///
    /// # Panics
    ///
    /// If a CUDA call fails.
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

    /// `crate::gpu`'s `run` on CUDA, panicking on a CUDA failure.
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
        self.try_run(ws, view, options, prior, start, planar, restorer)
            .unwrap_or_else(|e| panic!("CUDA PatchMatch on view {view}: {e}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_run(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        prior: Option<&[DepthMap]>,
        start: Option<&DepthMap>,
        planar: Option<(&PlanarPriorOptions, &PlanarPrior)>,
        restorer: DetailRestorer,
    ) -> Result<PassOutput, CudaError> {
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
        // The sources' earlier depth maps back to back, at their offsets.
        let mut priors = Vec::with_capacity(prior_len);
        if let Some(maps) = prior {
            for &s in &source_ids {
                priors.extend_from_slice(&maps[s].depth);
            }
        }
        let image = ws.views[view].gray_levels();
        let inputs = Inputs {
            width: w as u32,
            height: h as u32,
            num_sources: sources.len() as u32,
            texture_width: texture_width as u32,
            texture_height: texture_height as u32,
            sources: sources.as_ptr(),
            layers: layers.as_ptr(),
            image: image.as_ptr(),
            prior_len: priors.len() as u32,
            priors: priors.as_ptr(),
            init_planes: init_planes.as_ptr(),
            prior_planes: planar.map_or(std::ptr::null(), |(_, p)| p.planes.as_ptr()),
            pre_costs: u32::from(restorer != DetailRestorer::Off),
            classes: u32::from(deformable.is_some() || sort.is_some()),
            deformable: u32::from(deformable.is_some()),
        };
        if let Some((_, p)) = planar {
            assert_eq!(p.planes.len(), w * h, "prior planes");
        }

        let context = self.context.lock().unwrap_or_else(PoisonError::into_inner);
        let ctx = context.0.as_ptr();
        // SAFETY (every call below): the context is live and held through
        // the lock; the inputs point at buffers of the sizes `begin` reads
        // for this pass (`PassSetup`'s), which outlive the call, and the
        // outputs hold `w * h` elements, as the pass's buffers do.
        self.check(ctx, unsafe { (self.api.begin)(ctx, &inputs, &base) })?;
        drop((layers, init_planes, priors));
        // `classify` on the current planes with `params`.
        let classify = |params: &Params| -> Result<Vec<Class>, CudaError> {
            let mut codes = vec![0u32; w * h];
            self.check(ctx, unsafe {
                (self.api.classify)(ctx, params, codes.as_mut_ptr())
            })?;
            Ok(codes
                .iter()
                .map(|&c| match c {
                    0 => Class::Weak,
                    1 => Class::Strong,
                    _ => Class::Unknown,
                })
                .collect())
        };
        let mut pass_classes = None;
        if let Some(d) = &deformable {
            // The start's classes if the pass that made it sorted them,
            // else the start's pixels sorted now; each pixel's nearest
            // strong pixel on the CPU, then the anchors.
            let start_classes = start.map_or(&[][..], |map| &map.classes[..]);
            let classes = if start_classes.len() == w * h {
                let codes: Vec<u32> = start_classes.iter().map(|&c| class_code(c)).collect();
                self.check(ctx, unsafe { (self.api.set_classes)(ctx, codes.as_ptr()) })?;
                start_classes.to_vec()
            } else {
                classify(&base)?
            };
            let nearest = deformable::nearest_strong(&classes, w, h, d.nearest_radius);
            pass_classes = Some(classes);
            self.check(ctx, unsafe {
                (self.api.find_anchors)(ctx, &base, nearest.as_ptr())
            })?;
        }
        for params in &propagate {
            self.check(ctx, unsafe { (self.api.propagate)(ctx, params) })?;
        }

        let classes = match &sort {
            Some((_, params)) => classify(params)?,
            None => Vec::new(),
        };
        let mut planes = vec![[0.0f32; 4]; w * h];
        let mut costs = vec![0.0f32; w * h];
        self.check(ctx, unsafe {
            (self.api.read)(ctx, planes.as_mut_ptr(), costs.as_mut_ptr())
        })?;
        Ok((planes, costs, classes, pass_classes))
    }
}

impl Drop for CudaPatchMatch {
    fn drop(&mut self) {
        let context = self
            .context
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: the context came from `create` and is destroyed exactly
        // once, before `_library` (declared last) is dropped.
        unsafe { (self.api.destroy)(context.0.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_setup::checks;

    checks::impl_device!(CudaPatchMatch);

    /// The library named by [`LIBRARY_ENV`]; the tests skip without it.
    fn cuda() -> Option<CudaPatchMatch> {
        let Some(cuda) = CudaPatchMatch::from_env() else {
            eprintln!("{LIBRARY_ENV} unset; skipping");
            return None;
        };
        Some(cuda.expect("loading the CUDA MVS library"))
    }

    #[test]
    fn cuda_prior_fills_a_textureless_region() {
        if let Some(cuda) = cuda() {
            checks::prior_fills_a_textureless_region(&cuda);
        }
    }

    #[test]
    fn cuda_patch_match_recovers_a_plane() {
        if let Some(cuda) = cuda() {
            checks::recovers_a_plane(&cuda);
        }
    }

    #[test]
    fn cuda_restorer_matches_the_cpu() {
        if let Some(cuda) = cuda() {
            checks::restorer_matches_the_cpu(&cuda);
        }
    }

    #[test]
    fn cuda_deformable_patches_fill_a_textureless_region() {
        if let Some(cuda) = cuda() {
            checks::deformable_patches_fill_a_textureless_region(&cuda);
        }
    }

    #[test]
    fn cuda_strong_median_follows_the_pass_classes() {
        if let Some(cuda) = cuda() {
            checks::strong_median_follows_the_pass_classes(&cuda);
        }
    }

    #[test]
    fn cuda_deformable_passes_take_the_start_classes() {
        if let Some(cuda) = cuda() {
            checks::deformable_passes_take_the_start_classes(&cuda);
        }
    }

    /// Sampling the sources in f32 like the CPU, CUDA's photometric score
    /// matches the CPU's up to rounding.
    #[test]
    fn cuda_photometric_score_matches_the_cpu() {
        if let Some(cuda) = cuda() {
            checks::photometric_score_matches_the_cpu(&cuda, 1e-4, 1e-3);
        }
    }
}
