//! Train a 3DGS scene from a COLMAP dataset with the CUDA trainer (brush's
//! strategy and learning rates; `gsplat_train --strategy brush` on wgpu).
//!
//! ```text
//! scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda
//! VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so \
//!   cargo run --release -p visloc-gsplat-cuda --example gsplat_train_cuda -- \
//!   --data <colmap_root> --steps 30000 --out scene.ply [--eval-every 5000] \
//!   [--max-gaussians 2500000] [--sh-interval 1000] [--no-densify] [--profile] \
//!   [--sparse-sh-adam] [--bf16-moments] [--holdout 8] [--mesh mesh.ply [--planar-depth]] \
//!   [--depth-prior <dir> [--depth-weight 0.1 [--depth-weight-final W]]] \
//!   [--normal-weight 0.005 [--normal-start 7000]] [--alpha-mask] [--save-eval <dir>] \
//!   [--labels <dir> [--semantic-weight 0.02] [--semantic-lr 0.01] \
//!     [--semantic-logits-only | --semantic-shape]]
//! ```
//!
//! Initialises from `sparse/0/points3D`, holds out every `--holdout`th view
//! (8: the brush / Inria split; 0 trains on every view) and reports eval
//! PSNR and SSIM of the CUDA renders with `visloc_gsplat_train::metrics`.
//! Training time leaves the evaluations out. `--mesh` also extracts a mesh
//! from every view after training, as `gsplat_mesh_cuda` with its defaults.
//! `--depth-prior` adds a depth loss against `<dir>/<image name>.bin` (e.g.
//! `visloc-mvs --save-fused-depth`'s `fused_depth`): `--depth-weight` x the
//! mean relative error of the expected depth where the prior has one;
//! `--depth-weight-final` moves the weight log-linearly to W over the run.
//! `--normal-weight` adds the depth-normal consistency loss from
//! `--normal-start` on (the wgpu trainer's `normal_weight` / `normal_start`).
//! `--alpha-mask` leaves the pixels with image alpha below 128 out of the
//! loss (`gsplat_photos_cuda --mask-black`'s masked regions) and of the eval
//! PSNR; SSIM takes the image's values there. `--save-eval` writes the final
//! renders of the held-out views to `<dir>/<image name>.png`.
//! `--labels` trains a logit per class and gaussian against the training
//! views' labels (`scripts/sam3_labels.py`'s `<dir>/<image name>.png` and
//! `classes.txt`): `--semantic-weight` x the mean cross-entropy over the
//! labelled pixels. It also moves the gaussians' opacities and centres:
//! `--semantic-logits-only` trains the logits alone, `--semantic-shape` moves
//! the shapes too. The output then has the float properties
//! `class_logit_<c>` per class, which `gsplat_semantic_cuda --logits` reads.

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::imageops::FilterType;
use visloc_gsplat_core::ply::{save_ply, save_ply_with};
use visloc_gsplat_core::semantic::{parse_classes, NO_LABEL};
use visloc_gsplat_cuda::trainer::{CudaTrainer, SemanticTraining};
use visloc_gsplat_cuda::{CudaGsplat, LabelGeometry};
use visloc_gsplat_train::config::TrainConfig;
use visloc_gsplat_train::dataset::{
    find_model_dir, load_colmap_dataset_with, load_view_rgb, load_view_rgb_mask, EquirectOptions,
    RgbAndMask,
};
use visloc_gsplat_train::init::{read_points3d, seed_scene};
use visloc_gsplat_train::mesh::MeshOptions;
use visloc_gsplat_train::metrics::{psnr_masked, ssim};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut data: Option<PathBuf> = None;
    let mut out = PathBuf::from("gsplat_train_cuda.ply");
    let mut cfg = TrainConfig::brush_preset();
    let mut eval_every = 5000usize;
    let mut profile = false;
    let mut sparse_sh_adam = false;
    let mut bf16_moments = false;
    let mut holdout = 8usize;
    let mut mesh_out: Option<PathBuf> = None;
    let mut planar = false;
    let mut depth_prior: Option<PathBuf> = None;
    let mut depth_weight = 0.1f32;
    let mut depth_weight_final: Option<f32> = None;
    let mut save_eval: Option<PathBuf> = None;
    let mut labels: Option<PathBuf> = None;
    let mut semantic = SemanticTraining::default();
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
            // Adam moments in bf16, rounded stochastically.
            "--bf16-moments" => bf16_moments = true,
            "--holdout" => holdout = next()?.parse()?,
            "--mesh" => mesh_out = Some(PathBuf::from(next()?)),
            "--planar-depth" => planar = true,
            "--depth-prior" => depth_prior = Some(PathBuf::from(next()?)),
            "--depth-weight" => depth_weight = next()?.parse()?,
            "--depth-weight-final" => depth_weight_final = Some(next()?.parse()?),
            "--normal-weight" => cfg.normal_weight = next()?.parse()?,
            "--normal-start" => cfg.normal_start = next()?.parse()?,
            "--alpha-mask" => cfg.alpha_mask = true,
            "--save-eval" => save_eval = Some(PathBuf::from(next()?)),
            "--labels" => labels = Some(PathBuf::from(next()?)),
            "--semantic-weight" => semantic.weight = next()?.parse()?,
            "--semantic-lr" => semantic.lr = next()?.parse()?,
            "--semantic-logits-only" => semantic.geometry = LabelGeometry::LogitsOnly,
            "--semantic-shape" => semantic.geometry = LabelGeometry::Full,
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
    let eval_gt: Vec<RgbAndMask> = dataset
        .eval
        .iter()
        .map(|v| {
            if cfg.alpha_mask {
                load_view_rgb_mask(v)
            } else {
                load_view_rgb(v).map(|rgb| (rgb, None))
            }
        })
        .collect::<Result<_, _>>()?;

    let gs = CudaGsplat::from_env().ok_or(
        "set VISLOC_GSPLAT_CUDA_LIB to the library built by scripts/build_gsplat_cuda.sh",
    )??;
    let t0 = Instant::now();
    let mut trainer = CudaTrainer::new(gs, &dataset, &init, cfg.clone())?;
    trainer.set_sparse_sh_adam(sparse_sh_adam);
    trainer.set_bf16_moments(bf16_moments);
    if let Some(dir) = &depth_prior {
        let found = trainer.load_depth_priors(dir, depth_weight)?;
        trainer.set_depth_weight_final(depth_weight_final);
        println!(
            "depth priors for {found} / {} training views (weight {depth_weight})",
            dataset.train.len()
        );
    }
    let classes = match &labels {
        Some(dir) => {
            let classes = parse_classes(&std::fs::read_to_string(dir.join("classes.txt"))?)?;
            // Each training view's labels at its size (none where a view has
            // no label image).
            let mut bytes = Vec::new();
            let mut found = 0usize;
            for v in &dataset.train {
                let (w, h) = (v.camera.camera.width, v.camera.camera.height);
                let path = dir.join(Path::new(&v.name).with_extension("png"));
                if path.exists() {
                    let mut img = image::open(&path)?.to_luma8();
                    if img.dimensions() != (w, h) {
                        img = image::imageops::resize(&img, w, h, FilterType::Nearest);
                    }
                    bytes.extend_from_slice(img.as_raw());
                    found += 1;
                } else {
                    bytes.extend(std::iter::repeat_n(NO_LABEL, (w * h) as usize));
                }
            }
            trainer.set_labels(classes.len(), &bytes, semantic)?;
            println!(
                "labels for {found} / {} training views, {} classes (weight {}, lr {}{})",
                dataset.train.len(),
                classes.len(),
                semantic.weight,
                semantic.lr,
                match semantic.geometry {
                    LabelGeometry::LogitsOnly => ", logits only",
                    LabelGeometry::OpacityPosition => "",
                    LabelGeometry::Full => ", shapes too",
                }
            );
            classes
        }
        None => Vec::new(),
    };
    println!(
        "setup {:.1}s, scene extent {:.3}",
        t0.elapsed().as_secs_f64(),
        trainer.extent()
    );

    let eval = |trainer: &mut CudaTrainer| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let (mut p, mut s) = (0.0, 0.0);
        for (v, (gt, valid)) in dataset.eval.iter().zip(&eval_gt) {
            let mut img = trainer.render(&v.camera)?;
            let cam = &v.camera.camera;
            p += psnr_masked(&img, gt, valid.as_deref());
            // SSIM has no mask: the masked pixels take the image's values.
            if let Some(valid) = valid {
                for ((px, g), &ok) in img.iter_mut().zip(gt).zip(valid) {
                    if !ok {
                        *px = *g;
                    }
                }
            }
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
                let normal = if cfg.normal_weight > 0.0 {
                    format!("  normal {:.4}", trainer.last_normal_loss())
                } else {
                    String::new()
                };
                let labels = if classes.is_empty() {
                    String::new()
                } else {
                    format!("  labels {:.4}", trainer.last_semantic_loss())
                };
                println!(
                    "step {step:6}  l1 {loss:.4}{depth}{normal}{labels}{quality}  gaussians {}  ({train_s:.0}s)",
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
    match trainer.logits()? {
        Some(logits) => {
            // One property per class: class_logit_<c>.
            let k = classes.len();
            let columns: Vec<Vec<f32>> = (0..k)
                .map(|c| logits.iter().skip(c).step_by(k).copied().collect())
                .collect();
            let names: Vec<String> = (0..k).map(|c| format!("class_logit_{c}")).collect();
            let extra: Vec<(&str, &[f32])> = names
                .iter()
                .zip(&columns)
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            save_ply_with(&trainer.scene()?, &extra, &out)?;
        }
        None => save_ply(&trainer.scene()?, &out)?,
    }
    println!("wrote {}", out.display());
    if let Some(dir) = &save_eval {
        std::fs::create_dir_all(dir)?;
        for v in &dataset.eval {
            let cam = &v.camera.camera;
            let bytes: Vec<u8> = trainer
                .render(&v.camera)?
                .iter()
                .flatten()
                .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            let name = Path::new(&v.name).with_extension("png");
            let path = dir.join(name.file_name().ok_or("eval view without a name")?);
            image::save_buffer(&path, &bytes, cam.width, cam.height, image::ColorType::Rgb8)?;
        }
        println!(
            "wrote {} eval renders to {}",
            dataset.eval.len(),
            dir.display()
        );
    }
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
