//! Mesh extraction from the scene on the device: the CUDA median depth
//! ([`CudaGsplat::render_depth`]) fused by
//! [`visloc_gsplat_train::mesh::fuse_views`], as the wgpu
//! `extract_from_splat`.

use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_train::mesh::{fuse_views, Mesh, MeshOptions};

use crate::{CudaGsplat, GsplatCudaError};

/// Render the current scene's median depth (SH up to `sh_degree`; at the
/// splat planes when `planar`, see [`CudaGsplat::render_depth`]) from every
/// view, fuse, extract, filter by `support` and drop small components.
/// Returns the mesh and a one-line summary.
pub fn extract_mesh(
    gs: &mut CudaGsplat,
    sh_degree: u32,
    planar: bool,
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
) -> Result<(Mesh, String), GsplatCudaError> {
    fuse_views(views, support, opts, |view| {
        let frame = gs.render_depth(view, [0.0; 3], sh_degree, planar)?;
        let rgb = frame
            .rgb
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Ok((rgb, frame.depth))
    })
}
