//! Train a 3DGS scene on the CPU with Adam or the local Gauss-Newton step.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-cpu --example gsplat_cpu_train -- \
//!     --data <colmap root> --optimizer adam|newton --steps 7000 \
//!     [--eval-every 1000] [--loss l2|l1] [--neighbours 3] [--lambda 0.1] \
//!     [--sh-interval N] [--max-gaussians N] [--out scene.ply]
//! ```
//!
//! Every 8th view (by name) is held out; eval PSNR is over those. The
//! population is fixed (the SfM points, no densification).

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use visloc_gsplat_core::ply::save_ply;
use visloc_gsplat_cpu::model::Model;
use visloc_gsplat_cpu::train::{render, AdamConfig, Loss, NewtonConfig, TrainView, Trainer};
use visloc_gsplat_train::dataset::{load_colmap_dataset, load_view_rgb};
use visloc_gsplat_train::metrics::psnr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut data: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut optimizer = "adam".to_string();
    let mut steps = 7000usize;
    let mut eval_every = 1000usize;
    let mut loss = Loss::L2;
    let mut adam = AdamConfig::default();
    let mut newton = NewtonConfig::default();
    let mut sh_interval: Option<usize> = None;
    let mut max_gaussians: Option<usize> = None;
    let mut seed = 42u64;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--data" => data = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--optimizer" => optimizer = next()?,
            "--steps" => steps = next()?.parse()?,
            "--eval-every" => eval_every = next()?.parse()?,
            "--loss" => {
                loss = match next()?.as_str() {
                    "l2" => Loss::L2,
                    "l1" => Loss::L1,
                    o => return Err(format!("unknown loss {o}").into()),
                }
            }
            "--neighbours" => newton.neighbours = next()?.parse()?,
            "--neighbour-weight" => newton.neighbour_weight = next()?.parse()?,
            "--lambda" => newton.lambda = next()?.parse()?,
            "--max-dmean" => newton.max_dmean = next()?.parse()?,
            "--max-dlog-scale" => newton.max_dlog_scale = next()?.parse()?,
            "--max-dlogit" => newton.max_dlogit = next()?.parse()?,
            "--max-dcol" => newton.max_dcol = next()?.parse()?,
            "--sh-interval" => sh_interval = Some(next()?.parse()?),
            "--max-gaussians" => max_gaussians = Some(next()?.parse()?),
            "--seed" => seed = next()?.parse()?,
            o => return Err(format!("unknown argument {o}").into()),
        }
    }
    if let Some(s) = sh_interval {
        adam.sh_interval = s;
        newton.sh_interval = s;
    }
    let data = data.ok_or("--data is required")?;
    let t_setup = Instant::now();
    let ds = load_colmap_dataset(&data, Some(8))?;
    let mut scene = ds.init.clone();
    if let Some(m) = max_gaussians {
        scene.gaussians.truncate(m);
    }
    let model = Model::from_scene(&scene);
    let views: Vec<TrainView> = ds
        .train
        .par_iter()
        .map(|v| Ok(TrainView::new(v.camera, &load_view_rgb(v)?)))
        .collect::<Result<_, visloc_gsplat_train::dataset::DatasetError>>()?;
    let eval: Vec<_> = ds
        .eval
        .par_iter()
        .map(|v| Ok((v.camera, load_view_rgb(v)?)))
        .collect::<Result<Vec<_>, visloc_gsplat_train::dataset::DatasetError>>()?;
    let bg = [0.0f32; 3];
    let mut tr = Trainer::new(model, views, bg);
    tr.loss = loss;
    if optimizer == "newton" && loss != Loss::L2 {
        return Err("the Newton step needs --loss l2".into());
    }
    println!(
        "{} gaussians, {} train / {} eval views, {}x{}, extent {:.3}, setup {:.1}s, {} threads",
        tr.model.len(),
        tr.views.len(),
        eval.len(),
        tr.views[0].view.camera.width,
        tr.views[0].view.camera.height,
        tr.extent,
        t_setup.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    let eval_psnr = |tr: &Trainer| -> f64 {
        let mut s = 0.0;
        for (cam, gt) in &eval {
            let r = render(&tr.model, cam, bg);
            let img: Vec<[f32; 3]> = r
                .raster
                .image
                .chunks(3)
                .map(|c| [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)])
                .collect();
            s += psnr(&img, gt);
        }
        s / eval.len() as f64
    };
    println!("step      0  eval psnr {:.3}", eval_psnr(&tr));
    // A fixed shuffled order of views, reshuffled each epoch.
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut order: Vec<usize> = Vec::new();
    let t0 = Instant::now();
    let mut train_s = 0.0f64;
    let mut loss_sum = 0.0f64;
    let mut loss_n = 0usize;
    for step in 1..=steps {
        if order.is_empty() {
            order = (0..tr.views.len()).collect();
            for i in (1..order.len()).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = ((rng >> 33) as usize) % (i + 1);
                order.swap(i, j);
            }
        }
        let vi = order.pop().unwrap_or(0);
        let ts = Instant::now();
        let l = match optimizer.as_str() {
            "adam" => tr.adam_step(vi, step - 1, steps, &adam),
            "newton" => tr.newton_step(vi, step - 1, &newton),
            o => return Err(format!("unknown optimizer {o}").into()),
        };
        train_s += ts.elapsed().as_secs_f64();
        loss_sum += l;
        loss_n += 1;
        if step % eval_every == 0 || step == steps {
            let p = eval_psnr(&tr);
            println!(
                "step {step:6}  train loss {:.5}  eval psnr {p:.3}  ({train_s:.1}s train, {:.1} ms/step)",
                loss_sum / loss_n as f64,
                1e3 * train_s / step as f64
            );
            loss_sum = 0.0;
            loss_n = 0;
        }
    }
    let tm = tr.timing;
    let per = |x: f64| 1e3 * x / steps as f64;
    println!(
        "ms/step: project+bin {:.1}, forward {:.1}, loss {:.1}, backward {:.1}, lift {:.1}, update {:.1}; total {:.1}s",
        per(tm[0]),
        per(tm[1]),
        per(tm[2]),
        per(tm[3]),
        per(tm[4]),
        per(tm[5]),
        t0.elapsed().as_secs_f64()
    );
    if let Some(out) = out {
        save_ply(&tr.model.to_scene(), &out)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}
