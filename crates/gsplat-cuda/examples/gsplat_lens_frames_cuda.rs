//! Render a trained splat and its depth through each dataset view's own
//! lens, and through a lens swept from a pinhole to a view's fisheye: the
//! frames of `scripts/make_fisheye_gif.py`.
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_lens_frames_cuda -- \
//!   --ply scene.ply --data <colmap_root> --out <dir> [--every 4] \
//!   [--sweep <image name> [--sweep-frames 48] [--sweep-from 90]]
//! ```
//!
//! Every `--every`-th view (in name order) gives `<dir>/splat_<k>.png` and
//! `<dir>/depth_<k>.bin`, both through the view's lens; `<dir>/views.txt`
//! lists `k`, the image name and its path. The sweep renders the view named
//! by `--sweep` through `--sweep-frames` lenses (`<dir>/sweep_splat_<j>.png`,
//! `sweep_depth_<j>.bin`); `<dir>/sweep.txt` lists `j` and each frame's
//! horizontal field of view in degrees, after a `# <image name> <path>`
//! line. It starts at a pinhole `--sweep-from` degrees across and ends at
//! the view's own lens: an equidistant lens whose distortion starts as the
//! tangent's series (so it images as a pinhole) and blends into the view's,
//! while the focal length shrinks to the view's.
//!
//! A depth file holds the expected depth where the opacity exceeds 0.5 (0
//! elsewhere; smoother than the median depth), laid out as
//! `visloc-mvs --save-fused-depth` writes depth maps: width and height
//! (`u32`), then `f32` depths. Through a fisheye the depth is the distance
//! from the camera centre.

use std::error::Error;
use std::path::{Path, PathBuf};

use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::lens::{unproject, Lens, Projection, LENS_COEFFS};
use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::dataset::load_colmap_dataset;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut data: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut every = 4usize;
    let mut sweep: Option<String> = None;
    let mut sweep_frames = 48usize;
    let mut sweep_from = 90.0f32;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(next()?)),
            "--data" => data = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--every" => every = next()?.parse::<usize>()?.max(1),
            "--sweep" => sweep = Some(next()?),
            "--sweep-frames" => sweep_frames = next()?.parse::<usize>()?.max(2),
            "--sweep-from" => sweep_from = next()?.parse()?,
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
    let mut gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    gs.set_scene(&PackedScene::from_scene(&scene))?;
    let mut render = |cam: &CameraView, stem: String| -> Result<(), Box<dyn Error>> {
        let frame = gs.render_depth(cam, [0.0; 3], scene.sh_degree, false)?;
        save_rgb(&out.join(format!("{stem}.png")), cam, &frame.rgb)?;
        let stem = stem.replace("splat", "depth");
        let expected: Vec<f32> = frame
            .expected
            .iter()
            .zip(&frame.opacity)
            .map(|(&e, &a)| if a > 0.5 { e / a } else { 0.0 })
            .collect();
        save_depth(&out.join(format!("{stem}.bin")), cam, &expected)
    };

    let mut list = String::new();
    for (k, v) in views.iter().step_by(every).enumerate() {
        render(&v.camera, format!("splat_{k:04}"))?;
        list.push_str(&format!("{k} {} {}\n", v.name, v.image_path.display()));
    }
    std::fs::write(out.join("views.txt"), list)?;
    println!("{} views through their lenses", views.len().div_ceil(every));

    if let Some(name) = sweep {
        let v = views
            .iter()
            .find(|v| v.name == name)
            .ok_or(format!("no view named {name}"))?;
        if v.camera.camera.lens.projection != Projection::Equidistant {
            return Err("the sweep ends at a fisheye (equidistant) view".into());
        }
        let mut fovs = format!("# {} {}\n", v.name, v.image_path.display());
        for j in 0..sweep_frames {
            let t = j as f32 / (sweep_frames - 1) as f32;
            let cam = CameraView {
                camera: sweep_camera(&v.camera.camera, sweep_from, t),
                ..v.camera
            };
            render(&cam, format!("sweep_splat_{j:04}"))?;
            fovs.push_str(&format!("{j} {:.1}\n", horizontal_fov(&cam.camera)));
        }
        std::fs::write(out.join("sweep.txt"), fovs)?;
        println!("{sweep_frames} sweep frames of {name}");
    }
    Ok(())
}

/// The sweep's camera at `t` in [0, 1] (eased): an equidistant lens with
/// the tangent's series `θ (1 + θ²/3 + 2θ⁴/15 + 17θ⁶/315 + 62θ⁸/2835)` as
/// its distortion at the start (a pinhole `from_deg` across) blending into
/// `end`'s, the focal length moving geometrically to `end`'s.
fn sweep_camera(end: &PinholeCamera, from_deg: f32, t: f32) -> PinholeCamera {
    const TAN: [f32; 4] = [1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0];
    let e = t * t * (3.0 - 2.0 * t);
    let theta = (0.5 * from_deg).to_radians();
    let series = TAN
        .iter()
        .enumerate()
        .map(|(i, k)| k * theta.powi(2 * i as i32 + 2))
        .sum::<f32>();
    let fx0 = 0.5 * end.width as f32 / (theta * (1.0 + series));
    let fx = fx0.powf(1.0 - e) * end.fx.powf(e);
    let fy = fx * end.fy / end.fx;
    let mut coeffs = [0.0f32; LENS_COEFFS];
    for (i, c) in coeffs.iter_mut().enumerate() {
        let start = TAN.get(i).copied().unwrap_or(0.0);
        *c = start * (1.0 - e) + end.lens.coeffs[i] * e;
    }
    PinholeCamera::new(end.width, end.height, fx, fy, end.cx, end.cy)
        .with_lens(Lens::new(Projection::Equidistant, coeffs))
}

/// Degrees between the optical axis and the rays through the left and
/// right borders along the principal row, summed.
fn horizontal_fov(c: &PinholeCamera) -> f64 {
    let angle = |u: f32| {
        let m = [((u - c.cx) / c.fx) as f64, 0.0];
        unproject(&c.lens, m).map_or(0.0, |r| r[0].hypot(r[1]).atan2(r[2]))
    };
    (angle(0.0) + angle(c.width as f32)).to_degrees()
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
