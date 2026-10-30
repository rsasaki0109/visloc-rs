//! Time the CUDA rasterizer's stages on a trained scene.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_cuda_bench -- \
//!   --scene docs/euroc_splat/euroc_v101.splat [--width 1280 --height 720] \
//!   [--views 6] [--iters 20] [--fov-deg 70] [--data <colmap_root> [--sparse-sh-adam]]
//! ```
//!
//! With `--data` (e.g. the dataset a `.ply` was trained on) each iteration is
//! a training step with every learning rate at zero on the next image of the
//! COLMAP model, in order: the kernels and gradients of training, without
//! changing the scene. The image size replaces `--width` / `--height`.
//!
//! Without it the scene has no cameras: the views stand at the median of the
//! means and turn about the world y axis, and each iteration renders one view
//! and runs the backward pass with a fixed synthetic `d_image`, reading the
//! image and every gradient back (the GPU idles during those copies, which a
//! laptop GPU answers with lower clocks).
//!
//! The table holds each stage's median GPU time over all iterations (a CUDA
//! event pair around each stage).

use std::path::PathBuf;

use nalgebra::{Rotation3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_cuda::{CudaGsplat, GscCamera, StepParams};
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::config::pack_rgba8;
use visloc_gsplat_train::dataset::{load_colmap_dataset_with, load_view_rgb, EquirectOptions};

struct Args {
    scene: PathBuf,
    width: u32,
    height: u32,
    views: usize,
    iters: usize,
    fov_deg: f32,
    data: Option<PathBuf>,
    sparse_sh_adam: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        scene: PathBuf::new(),
        width: 1280,
        height: 720,
        views: 6,
        iters: 20,
        fov_deg: 70.0,
        data: None,
        sparse_sh_adam: false,
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
            "--data" => args.data = Some(PathBuf::from(value())),
            "--sparse-sh-adam" => args.sparse_sh_adam = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: gsplat_cuda_bench --scene <.ply|.splat> [--width W] [--height H] \
                     [--views N] [--iters N] [--fov-deg F] [--data <colmap_root> [--sparse-sh-adam]]"
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

    let mut cuda = CudaGsplat::from_env()
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
    let views: Vec<CameraView> = match &args.data {
        Some(root) => {
            let dataset =
                load_colmap_dataset_with(root, None, &EquirectOptions::default(), &mut |_| {})
                    .expect("load the COLMAP dataset");
            let cam = dataset
                .train
                .first()
                .expect("a training view")
                .camera
                .camera;
            let mut images = Vec::new();
            for v in &dataset.train {
                images.extend(pack_rgba8(&load_view_rgb(v).expect("load image"), None));
            }
            cuda.set_images(cam.width, cam.height, &images)
                .expect("upload images");
            dataset.train.iter().map(|v| v.camera).collect()
        }
        None => (0..args.views)
            .map(|v| {
                let yaw = v as f32 / args.views as f32 * std::f32::consts::TAU;
                // World-to-camera: rotate the world by -yaw about y, then
                // move the centre to the origin.
                let r = Rotation3::from_euler_angles(0.0, -yaw, 0.0).into_inner();
                CameraView::new(r, -(r * centre), camera)
            })
            .collect(),
    };
    let first = views.first().expect("at least one view").camera;
    println!("{} views of {}x{}", views.len(), first.width, first.height);
    let pixels = (first.width * first.height) as usize;
    let d_image: Vec<f32> = (0..pixels * 3)
        .map(|i| (i as f32 * 0.37).sin() * 1e-3)
        .collect();

    let mut stage_ms: Vec<(String, Vec<f32>)> = Vec::new();
    let mut isects = Vec::new();
    // Host wall time of each training step (it returns once the GPU is
    // done), which also counts the work between the timed stages.
    let mut step_ms = Vec::new();
    for it in 0..args.iters + 1 {
        let vi = it % views.len();
        let mut num_intersections = None;
        if args.data.is_some() {
            let step = StepParams {
                camera: GscCamera::from_view(&views[vi]),
                image: vi as u32,
                sh_active_degree: scene.sh_degree as i32,
                background: [0.0; 3],
                ssim_weight: 0.2,
                lr: [0.0; 6],
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-15,
                bc1: 0.1,
                bc2: 0.001,
                scale_aux_coef: 0.0,
                opacity_aux_coef: 0.0,
                accumulate_stats: 1,
                noise_seed: 0,
                noise_scale: 0.0,
                noise_max: 0.0,
                sparse_sh_adam: args.sparse_sh_adam as u32,
                depth_weight: 0.0,
                normal_weight: 0.0,
            };
            let t = std::time::Instant::now();
            cuda.train_step(&step).expect("train step");
            if it > 0 {
                step_ms.push(t.elapsed().as_secs_f32() * 1e3);
            }
        } else {
            let frame = cuda
                .render(&views[vi], [0.0; 3], scene.sh_degree)
                .expect("render");
            cuda.backward(&d_image).expect("backward");
            num_intersections = Some(frame.num_intersections as f32);
        }
        // The first iteration grows the buffers; leave it out.
        if it == 0 {
            continue;
        }
        isects.extend(num_intersections);
        for (k, (name, ms)) in cuda.stage_times().into_iter().enumerate() {
            if stage_ms.len() <= k {
                stage_ms.push((name, Vec::new()));
            }
            stage_ms[k].1.push(ms);
        }
    }
    println!("{} iterations over {} views", args.iters, views.len());
    if !isects.is_empty() {
        println!("median intersections {:.0}", median(&mut isects));
    }
    // Rasterizer forward and backward, and the training step's own stages.
    let mut forward = 0.0;
    let mut backward = 0.0;
    let mut other = 0.0;
    for (name, ms) in &mut stage_ms {
        // The loss and optimizer stages only run in a training step.
        if ms.iter().all(|&m| m == 0.0) {
            continue;
        }
        let m = median(ms);
        match name.as_str() {
            "loss" | "optimizer" => other += m,
            n if n.ends_with("_bwd") => backward += m,
            _ => forward += m,
        }
        println!("{name:<12} {m:8.3} ms");
    }
    println!("{:<12} {forward:8.3} ms", "forward");
    println!("{:<12} {backward:8.3} ms", "backward");
    if other > 0.0 {
        println!("{:<12} {other:8.3} ms", "loss + optim");
    }
    println!("{:<12} {:8.3} ms", "total", forward + backward + other);
    if !step_ms.is_empty() {
        println!("{:<12} {:8.3} ms", "step wall", median(&mut step_ms));
    }
}
