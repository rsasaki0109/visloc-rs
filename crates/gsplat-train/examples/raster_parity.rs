//! Diagnostics: shader vs host (rayon) rasterizer on a real scene.
//!
//! ```text
//! raster_parity --ply scene.ply --data <colmap_root> [--views 20]
//! ```
//!
//! Per training view: the forward image of both rasterizers, and the
//! screen-space gradients of the L1 loss's dL/dC from both backward passes
//! on the same frame (plus the shader run twice, whose float atomics set the
//! noise floor).

use std::path::PathBuf;

use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_render::{GpuContext, Renderer};
use visloc_gsplat_train::dataset::{load_colmap_dataset, load_view_rgb};

fn rel_l2(a: &[[f32; 10]], b: &[[f32; 10]], k: usize) -> (f64, f64) {
    let (mut d2, mut n2, mut worst, mut scale) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x[k] as f64, y[k] as f64);
        d2 += (x - y).powi(2);
        n2 += x * x;
        worst = worst.max((x - y).abs());
        scale = scale.max(x.abs());
    }
    ((d2 / n2.max(1e-300)).sqrt(), worst / scale.max(1e-300))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut ply, mut data, mut views) = (None, None, 20usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ply" => ply = args.next().map(PathBuf::from),
            "--data" => data = args.next().map(PathBuf::from),
            "--views" => views = args.next().ok_or("--views n")?.parse()?,
            o => return Err(format!("unknown argument {o}").into()),
        }
    }
    let scene = load_ply(ply.ok_or("--ply")?)?;
    let ds = load_colmap_dataset(&data.ok_or("--data")?, Some(8))?;
    let (w, h) = (
        ds.train[0].camera.camera.width,
        ds.train[0].camera.camera.height,
    );
    let mut r = Renderer::new(GpuContext::new()?, &scene, w, h)?;
    let bg = [0.0f32; 3];
    let names = [
        "du", "dv", "dA", "dB", "dC", "dopac", "dr", "dg", "db", "refine",
    ];
    let mut sums = vec![[0.0f64; 4]; 10];
    let (mut img_max, mut img_mean) = (0.0f32, 0.0f64);
    let step = (ds.train.len() / views).max(1);
    let mut n = 0;
    for v in ds.train.iter().step_by(step).take(views) {
        let gt = load_view_rgb(v)?;
        r.set_cpu_raster(false);
        let img_s = r.render(&v.camera, bg);
        let npix = img_s.rgb.len() as f32;
        let d: Vec<[f32; 3]> = img_s
            .rgb
            .iter()
            .zip(&gt)
            .map(|(a, b)| [0, 1, 2].map(|c| (a[c] - b[c]).signum() / (3.0 * npix)))
            .collect();
        let g_s = r.backward_screen(&d)?;
        let g_s2 = r.backward_screen(&d)?;
        r.set_cpu_raster(true);
        let g_h = r.backward_screen(&d)?;
        let img_h = r.render(&v.camera, bg);
        for (a, b) in img_s.rgb.iter().zip(&img_h.rgb) {
            for c in 0..3 {
                let e = (a[c] - b[c]).abs();
                img_max = img_max.max(e);
                img_mean += e as f64;
            }
        }
        for (k, sum) in sums.iter_mut().enumerate() {
            let (hs, hw) = rel_l2(&g_s, &g_h, k);
            let (ss, sw) = rel_l2(&g_s, &g_s2, k);
            sum[0] += hs;
            sum[1] = sum[1].max(hw);
            sum[2] += ss;
            sum[3] = sum[3].max(sw);
        }
        n += 1;
    }
    println!(
        "{n} views, {} gaussians: forward host vs shader: max abs {img_max:.2e}, mean abs {:.2e}",
        scene.len(),
        img_mean / (n as f64 * (w * h * 3) as f64)
    );
    println!("screen grads: rel L2 (mean over views) and worst |diff| / max|g| (over views)");
    println!(
        "{:<8} {:>14} {:>14} {:>14} {:>14}",
        "", "host-shader L2", "worst", "shader-shader L2", "worst"
    );
    for (name, sum) in names.iter().zip(&sums) {
        println!(
            "{:<8} {:>14.2e} {:>14.2e} {:>16.2e} {:>14.2e}",
            name,
            sum[0] / n as f64,
            sum[1],
            sum[2] / n as f64,
            sum[3]
        );
    }
    Ok(())
}
