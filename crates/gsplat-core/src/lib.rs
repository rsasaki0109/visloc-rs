//! CPU reference implementation for 3D Gaussian Splatting, built to consume
//! visloc-rs SfM output.
//!
//! This crate is the **stage-0** foundation of the Rust 3DGS effort described in
//! `docs/rust_3dgs_plan.md`: pure-scalar types, lossless `.ply` and lightweight
//! `.splat` IO, spherical-harmonics evaluation, a COLMAP-model → scene loader,
//! and a deliberately slow **reference rasterizer**. It is a separate crate
//! from `visloc-core`, mirroring the `visloc-basalt` boundary, so the
//! localization core never gains a graphics dependency.
//!
//! The reference renderer exists to be unambiguously correct: the future
//! wgpu/Burn GPU kernels are validated against it (per-pixel and per-gradient)
//! before they are allowed to train anything.
//!
//! # Example
//!
//! ```no_run
//! use visloc_gsplat_core::splat;
//!
//! let scene = splat::load_splat("scene.splat")?;
//! println!("{} gaussians", scene.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(unsafe_code)]

pub mod backward;
pub mod camera;
pub mod cpu_render;
pub mod gaussian;
pub mod lens;
#[cfg(feature = "mesh")]
pub mod mesh;
pub mod ply;
pub mod sh;
pub mod splat;

#[cfg(feature = "colmap-io")]
pub mod colmap_scene;

pub use camera::{CameraView, PinholeCamera};
pub use cpu_render::{render, Image};
pub use gaussian::{Gaussian, Scene};
pub use lens::{Lens, Projection};
pub use ply::{load_ply, save_ply, PlyError};
pub use sh::{eval_sh_color, SH_C0};
pub use splat::{load_splat, save_splat, SplatError};

#[cfg(feature = "colmap-io")]
pub use colmap_scene::{load_colmap_scene, scene_from_visual_map, ColmapScene, ColmapSceneError};
