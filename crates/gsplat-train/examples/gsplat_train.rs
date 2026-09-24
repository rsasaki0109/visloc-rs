//! Train a 3DGS scene from a COLMAP dataset with the Rust/wgpu trainer.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features gpu --example gsplat_train -- \
//!     --data <colmap_root> --steps 7000 --out scene.ply [--eval-every 1000]
//! ```
//!
//! Initialises from `sparse/0/points3D.txt`, holds out every 8th view (the
//! brush / Inria split), trains on the rest and reports eval PSNR with the
//! same renderer and metric as `gsplat_eval`.

use std::path::PathBuf;
use std::time::Instant;

use visloc_gsplat_core::ply::save_ply;
use visloc_gsplat_render::GpuContext;
use visloc_gsplat_train::dataset::{load_colmap_dataset, load_view_rgb};
use visloc_gsplat_train::init::{read_points3d_txt, seed_scene};
use visloc_gsplat_train::metrics::psnr;
use visloc_gsplat_train::trainer::{TrainConfig, Trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut data: Option<PathBuf> = None;
    let mut out = PathBuf::from("gsplat_train.ply");
    let mut cfg = TrainConfig::default();
    let mut eval_every = 1000usize;
    let mut export_steps: Vec<usize> = Vec::new();
    let mut init_ply: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--data" => data = Some(PathBuf::from(next()?)),
            "--out" => out = PathBuf::from(next()?),
            "--steps" => cfg.steps = next()?.parse()?,
            "--eval-every" => eval_every = next()?.parse()?,
            "--no-densify" => cfg.densify = None,
            "--max-gaussians" => {
                let cap = next()?.parse()?;
                if let Some(d) = cfg.densify.as_mut() {
                    d.max_gaussians = cap;
                }
            }
            "--init-ply" => init_ply = Some(PathBuf::from(next()?)),
            "--export-steps" => {
                export_steps = next()?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let data = data.ok_or("--data <colmap root> is required")?;

    let dataset = load_colmap_dataset(&data, Some(8))?;
    let model = [data.join("sparse").join("0"), data.join("sparse")]
        .into_iter()
        .find(|d| d.join("points3D.txt").exists())
        .ok_or("no sparse/0/points3D.txt")?;
    let points = read_points3d_txt(model.join("points3D.txt"))?;
    let init = match &init_ply {
        Some(p) => visloc_gsplat_core::ply::load_ply(p)?,
        None => seed_scene(&points, 3),
    };
    println!(
        "{} points, {} train / {} eval views",
        points.len(),
        dataset.train.len(),
        dataset.eval.len()
    );
    let eval_gt: Vec<Vec<[f32; 3]>> = dataset
        .eval
        .iter()
        .map(load_view_rgb)
        .collect::<Result<_, _>>()?;

    let ctx = GpuContext::new()?;
    println!("adapter: {}", ctx.adapter_info.name);
    let t0 = Instant::now();
    let mut trainer = Trainer::new(ctx, &dataset, &init, cfg.clone())?;
    println!(
        "setup {:.1}s, scene extent {:.3}",
        t0.elapsed().as_secs_f64(),
        trainer.extent()
    );

    let train_start = Instant::now();
    let eval = |trainer: &mut Trainer| {
        let mut sum = 0.0;
        for (v, gt) in dataset.eval.iter().zip(&eval_gt) {
            let img = trainer.render(&v.camera);
            sum += psnr(&img.rgb, gt);
        }
        sum / dataset.eval.len().max(1) as f64
    };
    for step in 1..=cfg.steps {
        trainer.step()?;
        if export_steps.contains(&step) {
            let p = out.with_file_name(format!(
                "{}_{step:05}.ply",
                out.file_stem().unwrap_or_default().to_string_lossy()
            ));
            save_ply(&trainer.scene(), &p)?;
            println!("  exported {}", p.display());
        }
        if let Some(r) = trainer.take_densify_report() {
            if step % 1000 == 0 {
                println!(
                    "  densify @{step}: {} -> {} (+{} clone, +{} split, -{} prune)",
                    r.before, r.after, r.cloned, r.split, r.pruned
                );
            }
        }
        if step % 1000 == 0 && std::env::var_os("GSPLAT_MEM_REPORT").is_some() {
            if let Some(r) = trainer.memory_report() {
                println!("  {r}");
            }
        }
        if step % 100 == 0 {
            let loss = trainer.take_mean_loss();
            if let Some(p) = trainer.take_profile() {
                let line: Vec<String> = p
                    .stages
                    .iter()
                    .map(|(n, ms, c)| format!("{n} {:.1}", ms / *c as f64))
                    .collect();
                println!("  profile (ms/step): {}", line.join("  "));
            }
            if step % eval_every == 0 || step == cfg.steps {
                let train_s = train_start.elapsed().as_secs_f64();
                let p = eval(&mut trainer);
                println!(
                    "step {step:6}  l1 {loss:.4}  eval psnr {p:.3}  gaussians {}  ({train_s:.0}s)",
                    trainer.num_gaussians()
                );
            } else if step % 500 == 0 {
                println!(
                    "step {step:6}  l1 {loss:.4}  gaussians {}",
                    trainer.num_gaussians()
                );
            }
        }
    }
    println!(
        "trained {} steps in {:.1}s",
        cfg.steps,
        train_start.elapsed().as_secs_f64()
    );
    save_ply(&trainer.scene(), &out)?;
    println!("wrote {}", out.display());
    Ok(())
}
