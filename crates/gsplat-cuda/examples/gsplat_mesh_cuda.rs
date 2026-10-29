//! Extract a coloured triangle mesh from a trained splat with the CUDA
//! renderer (`gsplat_mesh` on wgpu).
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_mesh_cuda -- \
//!   --ply scene.ply --data <colmap_root> --out mesh.ply \
//!   [--voxel V] [--trunc-voxels 4] [--max-depth D] [--min-component 500] \
//!   [--min-weight 3] [--no-carve] [--support-voxels 10] [--planar-depth]
//! ```
//!
//! Renders the median depth of the splat from every dataset view, fuses the
//! depths into a sparse TSDF and extracts the surface with surface nets, with
//! the same defaults and filters as `gsplat_mesh` (see
//! `visloc_gsplat_train::mesh`). `--planar-depth` takes each pixel's median
//! depth where its ray meets the plane of the splat that takes the
//! transmittance to 0.5, instead of that splat's centre depth.

use std::path::PathBuf;

use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_cuda::mesh::extract_mesh;
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::dataset::load_colmap_dataset;
use visloc_gsplat_train::mesh::{rig_scale, scene_scale, MeshOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut data: Option<PathBuf> = None;
    let mut out = PathBuf::from("mesh.ply");
    let mut opts = MeshOptions::default();
    let mut planar = false;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(next()?)),
            "--data" => data = Some(PathBuf::from(next()?)),
            "--out" => out = PathBuf::from(next()?),
            "--voxel" => opts.voxel = Some(next()?.parse()?),
            "--trunc-voxels" => opts.trunc_voxels = next()?.parse()?,
            "--max-depth" => opts.max_depth = Some(next()?.parse()?),
            "--min-component" => opts.min_component = next()?.parse()?,
            "--min-weight" => opts.min_weight = next()?.parse()?,
            "--no-carve" => opts.carve = false,
            "--support-voxels" => opts.support_voxels = next()?.parse()?,
            "--planar-depth" => planar = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <scene.ply> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;

    let scene = load_ply(&ply)?;
    let dataset = load_colmap_dataset(&data, None)?;
    let views: Vec<_> = dataset
        .train
        .iter()
        .chain(&dataset.eval)
        .map(|v| &v.camera)
        .collect();
    let support: Vec<[f32; 3]> = dataset
        .init
        .gaussians
        .iter()
        .map(|g| [g.mean.x, g.mean.y, g.mean.z])
        .collect();
    println!(
        "{} gaussians, {} views, {} SfM points; rig scale {:.3}, scene scale {:.3}",
        scene.len(),
        views.len(),
        support.len(),
        rig_scale(&views),
        scene_scale(&views, &support)
    );
    let mut gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    gs.set_scene(&PackedScene::from_scene(&scene))?;
    let (mesh, summary) = extract_mesh(&mut gs, scene.sh_degree, planar, &views, &support, &opts)?;
    println!("{summary}");
    mesh.write_ply(&out)?;
    println!("wrote {}", out.display());
    Ok(())
}
