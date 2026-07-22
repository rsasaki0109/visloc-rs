//! Runtime-loaded native CUDA DPVO correlation backend.
//!
//! The DLL is built explicitly with `scripts/build_dpvo_cuda_kernels.ps1`;
//! it is not compiled by Cargo, so ordinary builds and docs.rs do not require
//! a CUDA toolkit. The SHA-256-bound V4 runner owns the resulting artifact.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_float, c_int, c_void, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use libloading::Library;
use ndarray::{Array2, Array3, ArrayView4};

const FNET_DIM: usize = 128;
const PATCH: usize = 3;
const CORR_DIM: usize = 882;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type LastErrorFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type RunFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_float,
    *const *const c_float,
    *const *const c_float,
    *const c_float,
    *const i32,
    *mut c_float,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_float,
) -> c_int;

#[derive(Debug)]
pub enum NativeCudaCorrelationError {
    Load { path: PathBuf, message: String },
    AbiVersion(u32),
    NullContext,
    Shape(String),
    Runtime { code: i32, message: String },
}

impl fmt::Display for NativeCudaCorrelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, message } => {
                write!(
                    f,
                    "load native CUDA correlation {}: {message}",
                    path.display()
                )
            }
            Self::AbiVersion(version) => {
                write!(f, "native CUDA correlation ABI {version}, expected 1")
            }
            Self::NullContext => write!(f, "native CUDA correlation returned a null context"),
            Self::Shape(message) => write!(f, "native CUDA correlation shape: {message}"),
            Self::Runtime { code, message } => {
                write!(f, "native CUDA correlation failed ({code}): {message}")
            }
        }
    }
}

impl std::error::Error for NativeCudaCorrelationError {}

pub struct NativeCudaCorrelation {
    _library: Library,
    context: NonNull<c_void>,
    destroy: DestroyFn,
    last_error: LastErrorFn,
    run: RunFn,
}

impl fmt::Debug for NativeCudaCorrelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeCudaCorrelation")
            .finish_non_exhaustive()
    }
}

impl NativeCudaCorrelation {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, NativeCudaCorrelationError> {
        let path = path.as_ref();
        let load_error = |message: String| NativeCudaCorrelationError::Load {
            path: path.to_path_buf(),
            message,
        };
        // SAFETY: every symbol is checked immediately against the versioned
        // C ABI declared by native/dpvo_cuda/dpvo_corr.cu. The Library is
        // retained for at least as long as every copied function pointer.
        let library = unsafe { Library::new(path) }.map_err(|e| load_error(e.to_string()))?;
        let (abi_version, create, destroy, last_error, run): (
            AbiVersionFn,
            CreateFn,
            DestroyFn,
            LastErrorFn,
            RunFn,
        ) = unsafe {
            (
                *library
                    .get(b"visloc_dpvo_corr_abi_version\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_dpvo_corr_create\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_dpvo_corr_destroy\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_dpvo_corr_last_error\0")
                    .map_err(|e| load_error(e.to_string()))?,
                *library
                    .get(b"visloc_dpvo_corr_run\0")
                    .map_err(|e| load_error(e.to_string()))?,
            )
        };
        let version = unsafe { abi_version() };
        if version != 1 {
            return Err(NativeCudaCorrelationError::AbiVersion(version));
        }
        let context =
            NonNull::new(unsafe { create() }).ok_or(NativeCudaCorrelationError::NullContext)?;
        Ok(Self {
            _library: library,
            context,
            destroy,
            last_error,
            run,
        })
    }

    pub fn run(
        &mut self,
        anchors: ArrayView4<'_, f32>,
        level0_frames: &[&Array3<f32>],
        level1_frames: &[&Array3<f32>],
        coords: ArrayView4<'_, f32>,
        targets: &[i32],
    ) -> Result<(Array2<f32>, f32), NativeCudaCorrelationError> {
        let (edges, channels, patch_y, patch_x) = anchors.dim();
        if channels != FNET_DIM || patch_y != PATCH || patch_x != PATCH {
            return Err(NativeCudaCorrelationError::Shape(format!(
                "anchors {:?}, expected (E,{FNET_DIM},{PATCH},{PATCH})",
                anchors.dim()
            )));
        }
        if coords.dim() != (edges, PATCH, PATCH, 2) || targets.len() != edges {
            return Err(NativeCudaCorrelationError::Shape(format!(
                "coords {:?}, targets {}, edges {edges}",
                coords.dim(),
                targets.len()
            )));
        }
        if level0_frames.is_empty() || level0_frames.len() != level1_frames.len() {
            return Err(NativeCudaCorrelationError::Shape(
                "pyramid frame lists are empty or differ in length".into(),
            ));
        }
        if let Some((edge, target)) = targets
            .iter()
            .copied()
            .enumerate()
            .find(|(_, target)| *target < 0 || (*target as usize) >= level0_frames.len())
        {
            return Err(NativeCudaCorrelationError::Shape(format!(
                "target {target} at edge {edge} is outside 0..{}",
                level0_frames.len()
            )));
        }
        let (_, height0, width0) = level0_frames[0].dim();
        let (_, height1, width1) = level1_frames[0].dim();
        for (index, (level0, level1)) in level0_frames.iter().zip(level1_frames.iter()).enumerate()
        {
            if level0.dim() != (FNET_DIM, height0, width0)
                || level1.dim() != (FNET_DIM, height1, width1)
            {
                return Err(NativeCudaCorrelationError::Shape(format!(
                    "pyramid frame {index} has inconsistent dimensions"
                )));
            }
        }
        let anchors = anchors.as_slice().ok_or_else(|| {
            NativeCudaCorrelationError::Shape("anchors are not contiguous".into())
        })?;
        let coords = coords
            .as_slice()
            .ok_or_else(|| NativeCudaCorrelationError::Shape("coords are not contiguous".into()))?;
        let level0_pointers: Result<Vec<_>, _> = level0_frames
            .iter()
            .map(|frame| {
                frame.as_slice().map(|slice| slice.as_ptr()).ok_or_else(|| {
                    NativeCudaCorrelationError::Shape("level0 is not contiguous".into())
                })
            })
            .collect();
        let level1_pointers: Result<Vec<_>, _> = level1_frames
            .iter()
            .map(|frame| {
                frame.as_slice().map(|slice| slice.as_ptr()).ok_or_else(|| {
                    NativeCudaCorrelationError::Shape("level1 is not contiguous".into())
                })
            })
            .collect();
        let level0_pointers = level0_pointers?;
        let level1_pointers = level1_pointers?;
        let checked_c_int = |name: &str, value: usize| {
            c_int::try_from(value).map_err(|_| {
                NativeCudaCorrelationError::Shape(format!(
                    "{name}={value} exceeds the native ABI integer range"
                ))
            })
        };
        let edges_c = checked_c_int("edges", edges)?;
        let frames_c = checked_c_int("frames", level0_frames.len())?;
        let height0_c = checked_c_int("height0", height0)?;
        let width0_c = checked_c_int("width0", width0)?;
        let height1_c = checked_c_int("height1", height1)?;
        let width1_c = checked_c_int("width1", width1)?;
        let mut output = Array2::<f32>::zeros((edges, CORR_DIM));
        let mut device_elapsed_ms = 0.0_f32;
        let code = unsafe {
            (self.run)(
                self.context.as_ptr(),
                anchors.as_ptr(),
                level0_pointers.as_ptr(),
                level1_pointers.as_ptr(),
                coords.as_ptr(),
                targets.as_ptr(),
                output
                    .as_slice_mut()
                    .expect("owned Array2 is contiguous")
                    .as_mut_ptr(),
                edges_c,
                frames_c,
                FNET_DIM as c_int,
                PATCH as c_int,
                height0_c,
                width0_c,
                height1_c,
                width1_c,
                3,
                &mut device_elapsed_ms,
            )
        };
        if code != 0 {
            let pointer = unsafe { (self.last_error)(self.context.as_ptr()) };
            let message = if pointer.is_null() {
                "no native error string".to_string()
            } else {
                unsafe { CStr::from_ptr(pointer) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(NativeCudaCorrelationError::Runtime { code, message });
        }
        Ok((output, device_elapsed_ms))
    }
}

impl Drop for NativeCudaCorrelation {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context.as_ptr()) };
    }
}
