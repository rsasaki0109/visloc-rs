//! Train a 3DGS scene from a COLMAP dataset with the CUDA trainer (brush's
//! strategy and learning rates; `gsplat_train --strategy brush` on wgpu).
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_train_cuda -- \
//!   --data <colmap_root> --steps 30000 --out scene.ply [--eval-every 5000] \
//!   [--max-gaussians 2500000] [--sh-interval 1000] [--no-densify] [--profile] \
//!   [--sparse-sh-adam] [--holdout 8] [--mesh mesh.ply [--planar-depth]] \n//!   [--depth-prior <dir> [--depth-weight 0.1]]
//! ```
//!
//! Initialises from `sparse/0/points3D`, holds out every `--holdout`th view
//! (8: the brush / Inria split; 0 trains on every view) and reports eval
//! PSNR and SSIM of the CUDA renders with `visloc_gsplat_train::metrics`.
//! Training time leaves the evaluations out. `--mesh` also extracts a mesh
//! from every view after training, as `gsplat_mesh_cuda` with its defaults.
//! `--depth-prior` adds a depth loss against `<dir>/<image name>.bin` (e.g.
//! `visloc-mvs --save-fused-depth`'s `fused_depth`): `--depth-weight` x the
//! mean relative error of the expected depth where the prior has one.

use std::path::PathBuf;
use std::time::Instant;

use visloc_gsplat_core::ply::save_ply;
use visloc_gsplat_cuda::trainer::CudaTrainer;
use visloc_gsplat_cuda::CudaGsplat;
use visloc_gsplat_train::config::TrainConfig;
use visloc_gsplat_train::dataset::{
    find_model_dir, load_colmap_dataset_with, load_view_rgb, EquirectOptions,
};
use visloc_gsplat_train::init::{read_points3d, seed_scene};
use visloc_gsplat_train::mesh::MeshOptions;
use visloc_gsplat_train::metrics::{psnr, ssim};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut data: Option<PathBuf> = None;
    let mut out = PathBuf::from("gsplat_train_cuda.ply");
    let mut cfg = TrainConfig::brush_preset();
    let mut eval_every = 5000usize;
    let mut profile = false;
    let mut sparse_sh_adam = false;
    let mut holdout = 8usize;
    let mut mesh_out: Option<PathBuf> = None;
    let mut planar = false;
    let mut depth_prior: Option<PathBuf> = None;
    let mut depth_weight = 0.1f32;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--data" => data = Some(PathBuf::from(next()?)),
            "--out" => out = PathBuf::from(next()?),
            "--steps" => cfg.steps = next()?.parse()?,
            "--eval-every" => eval_every = next()?.parse()?,
            "--sh-interval" => cfg.sh_degree_interval = next()?.parse()?,
            "--max-gaussians" => {
                let cap = next()?.parse()?;
                if let Some(b) = cfg.brush_refine.as_mut() {
                    b.max_gaussians = cap;
                }
            }
            // A fixed population: no refine.
            "--no-densify" => cfg.brush_refine = None,
            "--profile" => profile = true,
            // SH Adam only on the gaussians each step blends.
            "--sparse-sh-adam" => sparse_sh_adam = true,
            "--holdout" => holdout = next()?.parse()?,
            "--mesh" => mesh_out = Some(PathBuf::from(next()?)),
            "--planar-depth" => planar = true,
            "--depth-prior" => depth_prior = Some(PathBuf::from(next()?)),
            "--depth-weight" => depth_weight = next()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let data = data.ok_or("--data <colmap root> is required")?;

    let dataset = load_colmap_dataset_with(
        &data,
        (holdout > 0).then_some(holdout),
        &EquirectOptions::default(),
        &mut |m| println!("{m}"),
    )?;
    let points = read_points3d(find_model_dir(&data)?)?;
    let init = seed_scene(&points, 3);
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

    let gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    let t0 = Instant::now();
    let mut trainer = CudaTrainer::new(gs, &dataset, &init, cfg.clone())?;
    trainer.set_sparse_sh_adam(sparse_sh_adam);
    if let Some(dir) = &depth_prior {
        let found = trainer.load_depth_priors(dir, depth_weight)?;
        println!(
            "depth priors for {found} / {} training views (weight {depth_weight})",
            dataset.train.len()
        );
    }
    println!(
        "setup {:.1}s, scene extent {:.3}",
        t0.elapsed().as_secs_f64(),
        trainer.extent()
    );

    let eval = |trainer: &mut CudaTrainer| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let (mut p, mut s) = (0.0, 0.0);
        for (v, gt) in dataset.eval.iter().zip(&eval_gt) {
            let img = trainer.render(&v.camera)?;
            let cam = &v.camera.camera;
            p += psnr(&img, gt);
            s += ssim(&img, gt, cam.width as usize, cam.height as usize);
        }
        let k = dataset.eval.len().max(1) as f64;
        Ok((p / k, s / k))
    };

    let mut train_s = 0.0f64;
    let mut stage_sum: Vec<(String, f64)> = Vec::new();
    let mut stage_steps = 0usize;
    let mut since = Instant::now();
    for step in 1..=cfg.steps {
        trainer.step()?;
        if profile {
            for (k, (name, ms)) in trainer.stage_times().into_iter().enumerate() {
                if stage_sum.len() <= k {
                    stage_sum.push((name, 0.0));
                }
                stage_sum[k].1 += ms as f64;
            }
            stage_steps += 1;
        }
        if let Some(r) = trainer.take_refine_report() {
            if step % 1000 == 0 {
                println!(
                    "  refine @{step}: {} -> {} (+{} grow, +{} replace, -{} prune)",
                    r.before, r.after, r.cloned, r.split, r.pruned
                );
            }
        }
        if step % 100 == 0 {
            let loss = trainer.take_mean_loss()?;
            if profile && step % 1000 == 0 {
                let line: Vec<String> = stage_sum
                    .iter()
                    .map(|(n, ms)| format!("{n} {:.2}", ms / stage_steps.max(1) as f64))
                    .collect();
                println!("  profile (ms/step): {}", line.join("  "));
                stage_sum.clear();
                stage_steps = 0;
            }
            if step % eval_every == 0 || step == cfg.steps {
                train_s += since.elapsed().as_secs_f64();
                let quality = if dataset.eval.is_empty() {
                    String::new()
                } else {
                    let (p, s) = eval(&mut trainer)?;
                    format!("  eval psnr {p:.3}  ssim {s:.4}")
                };
                let depth = if depth_prior.is_some() {
                    format!("  depth {:.4}", trainer.last_depth_loss())
                } else {
                    String::new()
                };
                println!(
                    "step {step:6}  l1 {loss:.4}{depth}{quality}  gaussians {}  ({train_s:.0}s)",
                    trainer.num_gaussians()
                );
                since = Instant::now();
            } else if step % 500 == 0 {
                println!(
                    "step {step:6}  l1 {loss:.4}  gaussians {}",
                    trainer.num_gaussians()
                );
            }
        }
    }
    println!(
        "trained {} steps in {train_s:.1}s (evaluations left out)",
        cfg.steps
    );
    save_ply(&trainer.scene()?, &out)?;
    println!("wrote {}", out.display());
    if let Some(path) = mesh_out {
        let views: Vec<_> = dataset
            .train
            .iter()
            .chain(&dataset.eval)
            .map(|v| &v.camera)
            .collect();
        let support: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z])
            .collect();
        let (mesh, summary) =
            trainer.extract_mesh(planar, &views, &support, &MeshOptions::default())?;
        println!("{summary}");
        mesh.write_ply(&path)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
