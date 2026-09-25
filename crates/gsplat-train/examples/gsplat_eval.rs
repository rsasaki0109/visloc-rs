//! Score a trained splat on a dataset's held-out views.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features gpu --example gsplat_eval -- \
//!     --ply export_30000.ply --data <colmap_root> [--eval-every 8] [--save-dir renders]
//! ```
//!
//! Renders every eval view (sorted by name, every Nth; brush / Inria split)
//! with the wgpu renderer on a black background and reports PSNR against the
//! ground-truth image. Any trainer's Inria-format `.ply` can be scored this
//! way, so methods are compared with one renderer, split and metric.

use std::path::PathBuf;

use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_render::{GpuContext, Renderer};
use visloc_gsplat_train::dataset::{load_colmap_dataset, load_view_rgb};
use visloc_gsplat_train::metrics::{psnr, ssim};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut data: Option<PathBuf> = None;
    let mut eval_every = 8usize;
    let mut save_dir: Option<PathBuf> = None;
    let mut sh_degree: Option<u32> = None;
    let mut sh_rest_coef_major = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ply" => ply = args.next().map(PathBuf::from),
            "--data" => data = args.next().map(PathBuf::from),
            "--eval-every" => {
                eval_every = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--eval-every needs a number")?
            }
            "--save-dir" => save_dir = args.next().map(PathBuf::from),
            // Diagnostics: evaluate SH only up to this degree.
            "--sh-degree" => sh_degree = args.next().and_then(|v| v.parse().ok()),
            // Diagnostics: read f_rest as coefficient-major ([k][ch]) instead
            // of the Inria channel-major ([ch][k]) layout.
            "--sh-rest-coef-major" => sh_rest_coef_major = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <path> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;

    let mut scene = load_ply(&ply)?;
    if sh_rest_coef_major {
        for g in scene.gaussians.iter_mut() {
            let rest = g.sh_rest.len() / 3;
            let orig = g.sh_rest.clone();
            for ch in 0..3 {
                for k in 0..rest {
                    g.sh_rest[ch * rest + k] = orig[k * 3 + ch];
                }
            }
        }
    }
    let dataset = load_colmap_dataset(&data, Some(eval_every))?;
    println!(
        "{} gaussians (sh degree {}), {} eval / {} train views",
        scene.len(),
        scene.sh_degree,
        dataset.eval.len(),
        dataset.train.len()
    );
    if let Some(dir) = &save_dir {
        std::fs::create_dir_all(dir)?;
    }

    let mut renderer: Option<(u32, u32, Renderer)> = None;
    let mut sum = 0.0f64;
    let mut ssim_sum = 0.0f64;
    let bg = [0.0, 0.0, 0.0];
    for view in &dataset.eval {
        let (w, h) = (view.camera.camera.width, view.camera.camera.height);
        // One renderer per resolution (datasets are usually single-camera).
        if renderer.as_ref().map(|r| (r.0, r.1)) != Some((w, h)) {
            let ctx = match renderer.take() {
                Some((_, _, r)) => r.ctx,
                None => GpuContext::new()?,
            };
            let mut r = Renderer::new(ctx, &scene, w, h)?;
            r.set_active_sh_degree(sh_degree);
            renderer = Some((w, h, r));
        }
        let (_, _, r) = renderer.as_mut().expect("renderer set above");
        let image = r.render(&view.camera, bg);
        let gt = load_view_rgb(view)?;
        let score = psnr(&image.rgb, &gt);
        let s = ssim(&image.rgb, &gt, w as usize, h as usize);
        ssim_sum += s;
        sum += score;
        println!("{:<24} psnr {score:6.2}  ssim {s:.4}", view.name);
        if let Some(dir) = &save_dir {
            let bytes: Vec<u8> = image
                .rgb
                .iter()
                .flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8))
                .collect();
            let stem = std::path::Path::new(&view.name)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| view.name.clone());
            image::save_buffer(
                dir.join(format!("{stem}.png")),
                &bytes,
                w,
                h,
                image::ColorType::Rgb8,
            )?;
        }
    }
    let n = dataset.eval.len().max(1) as f64;
    println!(
        "mean psnr {:.3}  ssim {:.4} over {} views",
        sum / n,
        ssim_sum / n,
        dataset.eval.len()
    );
    Ok(())
}
