//! A pure-CPU 3D Gaussian Splatting trainer (rayon over 16x16 tiles, 8-lane
//! rows that vectorize), with two optimizers on the same rasterizer:
//!
//! - Adam, the Inria recipe, one view per step;
//! - a per-gaussian local Gauss-Newton step after 3DGS² (Lan et al., arXiv
//!   2501.13975), which sums each gaussian's Gauss-Newton matrix over a
//!   primary view and its nearest neighbours and solves its own 11x11
//!   system, all gaussians at once.
//!
//! No GPU, wgpu or lavapipe: projection (with forward-mode dual numbers for
//! the Jacobian), binning, both rasterizer passes and the optimizers run on
//! the host.

// The kernels index several small fixed-size arrays with one loop variable
// (11x11 Cholesky, SH rows); iterator chains would read worse there.
#![allow(clippy::needless_range_loop)]

pub mod model;
pub mod project;
pub mod raster;
pub mod train;
