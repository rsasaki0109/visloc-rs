//! Time the CUDA rasterizer's stages on a trained scene.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_cuda_bench -- \
//!   --scene docs/euroc_splat/euroc_v101.splat [--width 1280 --height 720] \
//!   [--views 6] [--iters 20] [--fov-deg 70]
//! ```
//!
//! The scene has no cameras, so the views stand at the median of the means
//! and turn about the world y axis. Each iteration renders one view and runs
//! the backward pass with a fixed synthetic `d_image`; the table holds each
//! stage's median GPU time over all iterations (CUDA events, host copies
//! excluded).

use std::path::PathBuf;

use nalgebra::{Rotation3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_cuda::CudaRasterizer;
use visloc_gsplat_render::packing::PackedScene;

struct Args {
    scene: PathBuf,
    width: u32,
    height: u32,
    views: usize,
    iters: usize,
    fov_deg: f32,
}

fn parse_args() -> Args {
    let mut args = Args {
        scene: PathBuf::new(),
        width: 1280,
        height: 720,
        views: 6,
        iters: 20,
        fov_deg: 70.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--scene" => args.scene = PathBuf::from(value()),
            "--width" => args.width = value().parse().expect("--width"),
            "--height" => args.height = value().parse().expect("--height"),
            "--views" => args.views = value().parse().expect("--views"),
            "--iters" => args.iters = value().parse().expect("--iters"),
            "--fov-deg" => args.fov_deg = value().parse().expect("--fov-deg"),
            "-h" | "--help" => {
                eprintln!(
                    "usage: gsplat_cuda_bench --scene <.ply|.splat> [--width W] [--height H] \
                     [--views N] [--iters N] [--fov-deg F]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!args.scene.as_os_str().is_empty(), "--scene is required");
    args
}

fn load_scene(path: &PathBuf) -> Scene {
    match path.extension().and_then(|e| e.to_str()) {
        Some("splat") => visloc_gsplat_core::splat::load_splat(path).expect("load .splat"),
        _ => visloc_gsplat_core::ply::load_ply(path).expect("load .ply"),
    }
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let args = parse_args();
    let scene = load_scene(&args.scene);
    let mut coords: [Vec<f32>; 3] = Default::default();
    for g in &scene.gaussians {
        for (k, c) in coords.iter_mut().enumerate() {
            c.push(g.mean[k]);
        }
    }
    let centre = Vector3::new(
        median(&mut coords[0]),
        median(&mut coords[1]),
        median(&mut coords[2]),
    );
    println!(
        "{}: {} gaussians, SH degree {}, views at {:.2?}, {}x{}",
        args.scene.display(),
        scene.len(),
        scene.sh_degree,
        centre.as_slice(),
        args.width,
        args.height
    );

    let mut cuda = CudaRasterizer::from_env()
        .expect("set VISLOC_GSPLAT_CUDA_LIB to the built library")
        .expect("load the CUDA rasterizer");
    cuda.set_scene(&PackedScene::from_scene(&scene))
        .expect("upload");

    let f = 0.5 * args.width as f32 / (0.5 * args.fov_deg.to_radians()).tan();
    let camera = PinholeCamera::new(
        args.width,
        args.height,
        f,
        f,
        0.5 * args.width as f32,
        0.5 * args.height as f32,
    );
    let views: Vec<CameraView> = (0..args.views)
        .map(|v| {
            let yaw = v as f32 / args.views as f32 * std::f32::consts::TAU;
            // World-to-camera: rotate the world by -yaw about y, then move
            // the centre to the origin.
            let r = Rotation3::from_euler_angles(0.0, -yaw, 0.0).into_inner();
            CameraView::new(r, -(r * centre), camera)
        })
        .collect();
    let pixels = (args.width * args.height) as usize;
    let d_image: Vec<f32> = (0..pixels * 3)
        .map(|i| (i as f32 * 0.37).sin() * 1e-3)
        .collect();

    let mut stage_ms: Vec<(String, Vec<f32>)> = Vec::new();
    let mut isects = Vec::new();
    for it in 0..args.iters + 1 {
        let view = &views[it % views.len()];
        let frame = cuda
            .render(view, [0.0; 3], scene.sh_degree)
            .expect("render");
        cuda.backward(&d_image).expect("backward");
        // The first iteration grows the buffers; leave it out.
        if it == 0 {
            continue;
        }
        isects.push(frame.num_intersections as f32);
        for (k, (name, ms)) in cuda.stage_times().into_iter().enumerate() {
            if stage_ms.len() <= k {
                stage_ms.push((name, Vec::new()));
            }
            stage_ms[k].1.push(ms);
        }
    }
    println!(
        "median intersections {:.0} ({} iterations over {} views)",
        median(&mut isects),
        args.iters,
        views.len()
    );
    let mut forward = 0.0;
    let mut backward = 0.0;
    for (name, ms) in &mut stage_ms {
        let m = median(ms);
        if name.ends_with("_bwd") {
            backward += m;
        } else {
            forward += m;
        }
        println!("{name:<12} {m:8.3} ms");
    }
    println!("{:<12} {forward:8.3} ms", "forward");
    println!("{:<12} {backward:8.3} ms", "backward");
    println!("{:<12} {:8.3} ms", "total", forward + backward);
}
