//! Extract a coloured triangle mesh from a trained splat with the CUDA
//! renderer (`gsplat_mesh` on wgpu).
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_mesh_cuda -- \
//!   --ply scene.ply --data <colmap_root> --out mesh.ply \
//!   [--voxel V] [--trunc-voxels 4] [--max-depth D] [--min-component 500] \
//!   [--min-weight 3] [--no-carve] [--support-voxels 10] [--planar-depth] \
//!   [--expected-depth] [--gpu-tsdf [--slab-blocks 65536] [--depth-maps <dir> [--depth-maps-only]]]
//! ```
//!
//! Renders the median depth of the splat from every dataset view, fuses the
//! depths into a sparse TSDF and extracts the surface with surface nets, with
//! the same defaults and filters as `gsplat_mesh` (see
//! `visloc_gsplat_train::mesh`). `--planar-depth` takes each pixel's median
//! depth where its ray meets the plane of the splat that takes the
//! transmittance to 0.5, instead of that splat's centre depth;
//! `--expected-depth` the expected depth where the opacity exceeds 0.5 (the
//! channel the depth losses train). `--gpu-tsdf`
//! fuses and extracts on the device instead (`extract_mesh_gpu`), at most
//! `--slab-blocks` blocks of 8^3 voxels at a time. `--depth-maps <dir>` fuses
//! `<dir>/<image name>.bin` (visloc-mvs `--save-fused-depth`) in place of the
//! splat's depth wherever a map has one, and the splat's elsewhere
//! (`--depth-maps-only`: nowhere else).

use std::path::PathBuf;

use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_cuda::mesh::{
    extract_mesh, extract_mesh_gpu, read_depth_map, DepthMaps, MeshDepth,
};
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
    let mut depth = MeshDepth::Median;
    let mut gpu_tsdf = false;
    let mut slab_blocks = 0u32;
    let mut depth_maps: Option<PathBuf> = None;
    let mut maps_only = false;
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
            "--planar-depth" => depth = MeshDepth::Planar,
            "--expected-depth" => depth = MeshDepth::Expected,
            "--gpu-tsdf" => gpu_tsdf = true,
            "--slab-blocks" => slab_blocks = next()?.parse()?,
            "--depth-maps" => depth_maps = Some(PathBuf::from(next()?)),
            "--depth-maps-only" => maps_only = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <scene.ply> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;

    let scene = load_ply(&ply)?;
    let dataset = load_colmap_dataset(&data, None)?;
    let all: Vec<_> = dataset.train.iter().chain(&dataset.eval).collect();
    let views: Vec<_> = all.iter().map(|v| &v.camera).collect();
    // Per view `<dir>/<image name>.bin` at the view's size (0 where a view
    // has none).
    let maps: Option<Vec<f32>> = depth_maps.as_ref().map(|dir| {
        let mut found = 0;
        let mut maps = Vec::new();
        for v in &all {
            let (w, h) = (v.camera.camera.width, v.camera.camera.height);
            match read_depth_map(&dir.join(format!("{}.bin", v.name)), w, h) {
                Some(m) => {
                    found += 1;
                    maps.extend(m);
                }
                None => maps.extend(std::iter::repeat_n(0.0, (w * h) as usize)),
            }
        }
        println!("depth maps for {found} / {} views", all.len());
        maps
    });
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
    if maps.is_some() && !gpu_tsdf {
        return Err("--depth-maps needs --gpu-tsdf".into());
    }
    let (mesh, summary) = if gpu_tsdf {
        extract_mesh_gpu(
            &mut gs,
            scene.sh_degree,
            depth,
            &views,
            &support,
            &opts,
            slab_blocks,
            maps.as_deref().map(|m| DepthMaps {
                maps: m,
                fill_from_splat: !maps_only,
            }),
        )?
    } else {
        extract_mesh(&mut gs, scene.sh_degree, depth, &views, &support, &opts)?
    };
    println!("{summary}");
    mesh.write_ply(&out)?;
    println!("wrote {}", out.display());
    Ok(())
}
