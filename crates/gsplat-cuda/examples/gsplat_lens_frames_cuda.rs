//! Render a trained splat and its depth through the dataset's lens along
//! its camera path, smoothed between views: the frames of
//! `scripts/make_fisheye_gif.py`.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_lens_frames_cuda -- \
//!   --ply scene.ply --data <colmap_root> --out <dir> [--every 2] [--interp 4]
//! ```
//!
//! Every `--every`-th view (in name order) is a key; between two keys
//! `--interp` frames are rendered through the first key's camera (lens
//! included), its rotation interpolated spherically and its centre
//! linearly. Frame `j` gives `<dir>/splat_<j>.png` and `<dir>/depth_<j>.bin`;
//! `<dir>/views.txt` lists `j`, the nearer key's image name, the frame's
//! world-to-camera pose as COLMAP writes it (`qw qx qy qz tx ty tz`) and the
//! image's path.
//!
//! A depth file holds the expected depth where the opacity exceeds 0.5 (0
//! elsewhere; smoother than the median depth), laid out as
//! `visloc-mvs --save-fused-depth` writes depth maps: width and height
//! (`u32`), then `f32` depths. Through a fisheye the depth is the distance
//! from the camera centre.

use std::error::Error;
use std::path::{Path, PathBuf};

use nalgebra::{Rotation3, UnitQuaternion};
use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::dataset::load_colmap_dataset;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut data: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut every = 2usize;
    let mut interp = 4usize;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(next()?)),
            "--data" => data = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--every" => every = next()?.parse::<usize>()?.max(1),
            "--interp" => interp = next()?.parse::<usize>()?.max(1),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <scene.ply> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;
    let out = out.ok_or("--out <dir> is required")?;
    std::fs::create_dir_all(&out)?;

    let scene = load_ply(&ply)?;
    let dataset = load_colmap_dataset(&data, None)?;
    let mut views: Vec<_> = dataset.train.iter().chain(&dataset.eval).collect();
    views.sort_by(|a, b| a.name.cmp(&b.name));
    let keys: Vec<_> = views.iter().step_by(every).collect();
    let mut gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    gs.set_scene(&PackedScene::from_scene(&scene))?;

    let mut list = String::new();
    let mut j = 0usize;
    for (k, key) in keys.iter().enumerate() {
        let a = &key.camera;
        let b = keys.get(k + 1).map_or(a, |next| &next.camera);
        let quat = |v: &CameraView| {
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(v.rotation))
        };
        let (qa, qb) = (quat(a), quat(b));
        let (ca, cb) = (a.camera_center(), b.camera_center());
        let steps = if k + 1 < keys.len() { interp } else { 1 };
        for s in 0..steps {
            let t = s as f32 / interp as f32;
            let q = qa.try_slerp(&qb, t, 1e-6).unwrap_or(qa);
            let rotation = q.to_rotation_matrix().into_inner();
            let centre = ca.lerp(&cb, t);
            let cam = CameraView::new(rotation, -(rotation * centre), a.camera);
            let frame = gs.render_depth(&cam, [0.0; 3], scene.sh_degree, false)?;
            save_rgb(&out.join(format!("splat_{j:04}.png")), &cam, &frame.rgb)?;
            let expected: Vec<f32> = frame
                .expected
                .iter()
                .zip(&frame.opacity)
                .map(|(&e, &o)| if o > 0.5 { e / o } else { 0.0 })
                .collect();
            save_depth(&out.join(format!("depth_{j:04}.bin")), &cam, &expected)?;
            let near = if t < 0.5 { key } else { &keys[k + 1] };
            let tr = cam.translation;
            list.push_str(&format!(
                "{j} {} {} {} {} {} {} {} {} {}\n",
                near.name,
                q.w,
                q.i,
                q.j,
                q.k,
                tr.x,
                tr.y,
                tr.z,
                near.image_path.display()
            ));
            j += 1;
        }
    }
    std::fs::write(out.join("views.txt"), list)?;
    println!("{j} frames along {} keys", keys.len());
    Ok(())
}

fn save_rgb(path: &Path, view: &CameraView, rgb: &[f32]) -> Result<(), Box<dyn Error>> {
    let bytes: Vec<u8> = rgb
        .iter()
        .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::save_buffer(
        path,
        &bytes,
        view.camera.width,
        view.camera.height,
        image::ColorType::Rgb8,
    )?;
    Ok(())
}

fn save_depth(path: &Path, view: &CameraView, depth: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(8 + depth.len() * 4);
    bytes.extend(view.camera.width.to_le_bytes());
    bytes.extend(view.camera.height.to_le_bytes());
    for d in depth {
        bytes.extend(d.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}
