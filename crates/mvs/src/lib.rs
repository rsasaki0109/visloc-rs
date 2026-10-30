//! Multi-view stereo for visloc-rs: per-view depth and normal maps by
//! PatchMatch ([`patch_match`]), fused into a point cloud ([`fusion`]).
//!
//! The input is a COLMAP sparse model with pinhole cameras (undistorted
//! images) and its images ([`model`], [`workspace`]). Depth estimation is
//! ACMH with ACMP's planar prior ([`mod@planar_prior`]) and ACMM's geometric
//! consistency, on an image pyramid in ACMMP's order ([`pipeline`]; Xu and
//! Tao, CVPR 2019 and AAAI 2020; Xu et al., TPAMI 2022); fusion keeps the
//! points that enough source views agree on, and [`mod@mesh`] fuses their
//! depths into a TSDF mesh.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod deformable;
pub mod fusion;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod mesh;
pub mod model;
#[cfg(any(feature = "gpu", feature = "cuda"))]
mod pass_setup;
pub mod patch_match;
pub mod pipeline;
pub mod planar_prior;
pub mod ply;
pub mod workspace;

#[cfg(feature = "cuda")]
pub use cuda::CudaPatchMatch;
pub use deformable::Deformable;
pub use fusion::{fuse, fuse_with, FusedPoint, FusionOptions};
#[cfg(feature = "gpu")]
pub use gpu::GpuPatchMatch;
pub use mesh::{Mesh, MeshOptions, TsdfMesher};
pub use model::{read_text_model, SparseModel};
pub use patch_match::{
    estimate, estimate_from, estimate_with_prior, photometric_score, DepthMap, DetailRestorer,
    PatchMatchOptions, ViewSelection,
};
pub use pipeline::{
    geometric_pass, multi_scale_depth_maps, planar_prior_pass, Estimator, MultiScaleOptions,
    PriorEstimator, StartEstimator,
};
pub use planar_prior::{planar_prior, PlanarPrior, PlanarPriorOptions};
pub use ply::write_ply;
pub use workspace::{View, Workspace, WorkspaceOptions};
