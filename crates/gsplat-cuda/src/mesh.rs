//! Mesh extraction from the scene on the device: the CUDA median depth
//! ([`CudaGsplat::render_depth`]) fused by
//! [`visloc_gsplat_train::mesh::fuse_views`], as the wgpu
//! `extract_from_splat`, or fused on the device
//! ([`CudaGsplat::tsdf_surface`]) with the same options and filters.

use std::time::Instant;

use rayon::prelude::*;
use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_train::mesh::{fuse_views, scene_scale, Mesh, MeshOptions};

use crate::{CudaGsplat, GsplatCudaError, TsdfParams, TsdfSurface};

/// The per-pixel depth a mesh is fused from (see
/// [`CudaGsplat::render_depth`]); 0 is no surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshDepth {
    /// Camera z of the splat that takes the transmittance to 0.5.
    #[default]
    Median,
    /// Where the pixel's ray meets that splat's plane.
    Planar,
    /// The expected depth `sum_i T_i a_i z_i / (1 - T)` where the opacity
    /// `1 - T` exceeds 0.5: the channel the depth prior and the depth-normal
    /// loss train.
    Expected,
}

impl MeshDepth {
    /// [`MeshDepth::Planar`] if `planar`, else [`MeshDepth::Median`].
    pub fn median(planar: bool) -> Self {
        if planar {
            Self::Planar
        } else {
            Self::Median
        }
    }
}

/// Render the current scene's `depth` (SH up to `sh_degree`) from every
/// view, fuse, extract, filter by `support` and drop small components.
/// Returns the mesh and a one-line summary.
pub fn extract_mesh(
    gs: &mut CudaGsplat,
    sh_degree: u32,
    depth: MeshDepth,
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
) -> Result<(Mesh, String), GsplatCudaError> {
    fuse_views(views, support, opts, |view| {
        let frame = gs.render_depth(view, [0.0; 3], sh_degree, depth == MeshDepth::Planar)?;
        let rgb = frame
            .rgb
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let d = match depth {
            MeshDepth::Expected => frame
                .expected
                .iter()
                .zip(&frame.opacity)
                .map(|(&e, &a)| if a > 0.5 { e / a } else { 0.0 })
                .collect(),
            _ => frame.depth,
        };
        Ok((rgb, d))
    })
}

/// [`extract_mesh`] with the TSDF fused and extracted on the device
/// ([`CudaGsplat::tsdf_surface`], `slab_blocks` blocks at a time, 0: 65536;
/// the views are kept on the device up to 2 GiB): the same voxel size,
/// depth range, carving, minimum weight and filters. Unlike `fuse_views`,
/// every block sees every view.
pub fn extract_mesh_gpu(
    gs: &mut CudaGsplat,
    sh_degree: u32,
    depth: MeshDepth,
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
    slab_blocks: u32,
) -> Result<(Mesh, String), GsplatCudaError> {
    let scale = scene_scale(views, support);
    let voxel = opts.voxel.unwrap_or(scale / 256.0);
    let params = TsdfParams {
        voxel,
        trunc: voxel * opts.trunc_voxels,
        max_depth: opts.max_depth.unwrap_or(2.0 * scale),
        min_weight: opts.min_weight,
        carve: opts.carve as u32,
        free_space: 1,
        planar: (depth == MeshDepth::Planar) as u32,
        sh_active_degree: sh_degree as i32,
        slab_blocks,
        cache_mb: 2048,
        expected: (depth == MeshDepth::Expected) as u32,
    };
    let t0 = Instant::now();
    let surface = gs.tsdf_surface(views, &params)?;
    let fused = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut mesh = surface_mesh(&surface);
    let (blocks, slabs) = (surface.blocks, surface.slabs);
    drop(surface);
    let quads = t1.elapsed().as_secs_f64();
    let raw = mesh.triangles.len();
    let t2 = Instant::now();
    if opts.support_voxels > 0.0 && !support.is_empty() {
        mesh.keep_supported(support, opts.support_voxels * voxel);
    }
    mesh.remove_small_components(opts.min_component);
    let summary = format!(
        "mesh: {} views fused on the GPU in {fused:.1} s (voxel {voxel:.4}, {} blocks in {} slabs), quads {quads:.1} s, filters {:.1} s, {raw} -> {} triangles",
        views.len(),
        blocks,
        slabs,
        t2.elapsed().as_secs_f64(),
        mesh.triangles.len()
    );
    Ok((mesh, summary))
}

/// The mesh of a [`TsdfSurface`]: one quad (two triangles) per edge whose
/// four surrounding cells all have a vertex, facing the outside, as
/// `visloc_gsplat_core::mesh::Tsdf::extract`. The cells are looked up in a
/// sorted copy (tens of millions of them, four lookups per edge), in
/// parallel; the triangles keep the edges' order.
pub fn surface_mesh(s: &TsdfSurface) -> Mesh {
    let mut index: Vec<([i32; 3], u32)> = s
        .cells
        .par_iter()
        .enumerate()
        .map(|(i, c)| (*c, i as u32))
        .collect();
    index.par_sort_unstable_by_key(|e| e.0);
    let find = |k: [i32; 3]| {
        index
            .binary_search_by(|e| e.0.cmp(&k))
            .ok()
            .map(|i| index[i].1)
    };
    let quads: Vec<[[u32; 3]; 2]> = s
        .edges
        .par_iter()
        .filter_map(|e| {
            let v = [e[0], e[1], e[2]];
            let a = (e[3] & 3) as usize;
            // The four cells around the edge from v along a: v minus
            // {0, 1} along the two other axes.
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            let cell = |db: i32, dc: i32| {
                let mut k = v;
                k[b] -= db;
                k[c] -= dc;
                find(k)
            };
            let q = [cell(0, 0)?, cell(1, 0)?, cell(1, 1)?, cell(0, 1)?];
            // Outside (positive) -> inside: face toward the outside.
            let q = if e[3] & 4 != 0 {
                [q[0], q[3], q[2], q[1]]
            } else {
                q
            };
            Some([[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        })
        .collect();
    Mesh {
        vertices: s.positions.clone(),
        colors: s.colors.clone(),
        triangles: quads.into_iter().flatten().collect(),
    }
}
