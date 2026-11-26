//! Train a 3DGS scene on the CPU with Adam or the local Gauss-Newton step.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-cpu --example gsplat_cpu_train -- \
//!     --data <colmap root> [--optimizer adam|newton] [--steps 7000] \
//!     [--eval-every 1000] [--loss l1|l2] [--ssim-weight 0.2] [--no-densify] \
//!     [--max-gaussians N] [--sh-interval N] [--out scene.ply]
//! ```
//!
//! `--data` is a COLMAP root (`images/`, `sparse/0`), e.g. what
//! `gsplat_photos --steps 0 --out <dir>` leaves for undistorted pinhole
//! cameras. Every 8th view (by name) is held out; eval PSNR is over those. The
//! defaults are `gsplat_train`'s: L1 + 0.2 D-SSIM, Inria densification
//! (from step 500, every 100, opacity reset every 3000), SH degree up every
//! 1000 steps. `--strategy brush` trains with brush's refine instead
//! (`gsplat_train --strategy brush`, `gsplat_photos`'s default), its rates,
//! aux losses and mean noise. The Newton step (`--optimizer newton`, an experiment, see
//! `docs/gsplat_cpu_trainer.md`) needs `--loss l2 --no-densify`.

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use visloc_gsplat_core::ply::save_ply;
use visloc_gsplat_cpu::model::Model;
use visloc_gsplat_cpu::train::{render, AdamConfig, Loss, NewtonConfig, TrainView, Trainer};
use visloc_gsplat_train::dataset::{find_model_dir, load_colmap_dataset, load_view_rgb};
use visloc_gsplat_train::init::{read_points3d, seed_scene};
use visloc_gsplat_train::metrics::psnr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut data: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut optimizer = "adam".to_string();
    let mut steps = 7000usize;
    let mut eval_every = 1000usize;
    let mut loss = Loss::L1;
    let mut ssim_weight = 0.2f32;
    let mut densify_cfg = Some(visloc_gsplat_train::densify::DensifyConfig::default());
    let mut truncate: Option<usize> = None;
    let mut sparse_adam = true;
    let mut stride = 1usize;
    let mut res_schedule: Vec<f64> = Vec::new();
    let mut adam = AdamConfig::default();
    let mut brush = false;
    let mut newton = NewtonConfig::default();
    let mut sh_interval: Option<usize> = None;
    let mut max_gaussians: Option<usize> = None;
    let mut seed = 42u64;
    let mut overlap = 1.0f32;
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
            "--truncate" => truncate = Some(next()?.parse()?),
            "--ssim-weight" => ssim_weight = next()?.parse()?,
            "--no-densify" => densify_cfg = None,
            "--strategy" => match next()?.as_str() {
                "inria" => brush = false,
                "brush" => brush = true,
                o => return Err(format!("unknown strategy {o} (inria | brush)").into()),
            },
            "--dense-adam" => sparse_adam = false,
            "--stride" => stride = next()?.parse()?,
            // e.g. "0.2,0.5": a quarter of the resolution for the first 20%
            // of the steps, half until 50%, then full (coarse to fine).
            "--res-schedule" => {
                res_schedule = next()?
                    .split(',')
                    .map(|x| x.parse::<f64>())
                    .collect::<Result<_, _>>()?;
            }
            "--seed" => seed = next()?.parse()?,
            "--overlap" => overlap = next()?.parse()?,
            "--step" => newton.step = next()?.parse()?,
            "--decay" => newton.decay = next()?.parse()?,
            "--groups" => {
                let v = next()?;
                newton.groups =
                    ["mean", "shape", "opacity", "color"].map(|g| v.split(',').any(|x| x == g));
            }
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
    // The CPU projection is an undistorted pinhole's: refuse lenses it
    // would silently mis-project (gsplat_train renders those).
    if let Some(v) = ds
        .train
        .iter()
        .chain(&ds.eval)
        .find(|v| !v.camera.camera.lens.is_pinhole())
    {
        return Err(format!(
            "{}: a distorted or fisheye lens; gsplat_cpu_train handles undistorted pinholes only (use gsplat_train)",
            v.name
        )
        .into());
    }
    // As gsplat_train: SfM points, isotropic at the mean distance to 3 neighbours.
    let mut scene = seed_scene(&read_points3d(find_model_dir(&data)?)?, 3);
    if let Some(m) = truncate {
        scene.gaussians.truncate(m);
    }
    if let (Some(d), Some(m)) = (densify_cfg.as_mut(), max_gaussians) {
        d.max_gaussians = m;
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
    tr.overlap = overlap;
    tr.sparse_adam = sparse_adam;
    // brush's strategy (gsplat_train --strategy brush, gsplat_photos'
    // default): its refine replaces the Inria rule, with its rates and aux
    // losses.
    let brush_cfg = brush.then(|| visloc_gsplat_train::densify::BrushRefineConfig {
        max_gaussians: max_gaussians.unwrap_or(10_000_000),
        ..Default::default()
    });
    if brush {
        densify_cfg = None;
        let sh = adam.sh_interval;
        adam = AdamConfig {
            sh_interval: sh,
            ..AdamConfig::brush_preset()
        };
        tr.opac_loss_weight = 1e-9;
        tr.scale_loss_weight = 1e-8;
        tr.brush = brush_cfg.clone();
    }
    tr.stride = stride;
    tr.ssim_weight = if loss == Loss::L1 { ssim_weight } else { 0.0 };
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
                .map(|c| {
                    [
                        c[0].clamp(0.0, 1.0),
                        c[1].clamp(0.0, 1.0),
                        c[2].clamp(0.0, 1.0),
                    ]
                })
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
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = ((rng >> 33) as usize) % (i + 1);
                order.swap(i, j);
            }
        }
        let vi = order.pop().unwrap_or(0);
        // Coarse to fine: factor 2^(levels left) until each boundary.
        let frac = (step - 1) as f64 / steps as f64;
        let left = res_schedule.iter().filter(|&&b| frac < b).count();
        tr.res_factor = 1 << left;
        let ts = Instant::now();
        let l = match optimizer.as_str() {
            "adam" => tr.adam_step(vi, step - 1, steps, &adam),
            "newton" => tr.newton_step(vi, step - 1, &newton),
            o => return Err(format!("unknown optimizer {o}").into()),
        };
        if let Some(b) = brush_cfg.as_ref() {
            if step != steps && step % b.refine_every.max(1) == 0 {
                if let Some(rep) = tr.brush_refine(step) {
                    if step % 1000 == 0 {
                        println!(
                            "  refine @{step}: {} -> {} (+{} grown, +{} replaced, -{} pruned)",
                            rep.before, rep.after, rep.cloned, rep.split, rep.pruned
                        );
                    }
                }
            }
        }
        // Densify after the step, as gsplat_train does (not on the last).
        if let Some(d) = densify_cfg.as_ref() {
            let at_densify = step >= d.start && step <= d.stop && step % d.interval == 0;
            let at_reset = step < d.stop && step % d.opacity_reset_interval == 0;
            if step != steps && (at_densify || at_reset) {
                if let Some(rep) = tr.densify(
                    d,
                    at_densify,
                    at_reset,
                    step,
                    seed ^ (step as u64).wrapping_mul(0x9E37_79B9),
                ) {
                    if step % 500 == 0 {
                        println!(
                            "  densify @{step}: {} -> {} (+{} clone, +{} split, -{} prune)",
                            rep.before, rep.after, rep.cloned, rep.split, rep.pruned
                        );
                    }
                }
            }
        }
        train_s += ts.elapsed().as_secs_f64();
        loss_sum += l;
        loss_n += 1;
        if step % eval_every == 0 || step == steps {
            tr.sync();
            let p = eval_psnr(&tr);
            println!(
                "step {step:6}  train loss {:.5}  eval psnr {p:.3}  gaussians {}  ({train_s:.1}s train, {:.1} ms/step)",
                loss_sum / loss_n as f64,
                tr.model.len(),
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
    println!(
        "per view: {:.0} projected splats, {:.0} tile intersections",
        tr.splats as f64 / steps as f64,
        tr.isects as f64 / steps as f64
    );
    if let Some(out) = out {
        tr.sync();
        save_ply(&tr.model.to_scene(), &out)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}
