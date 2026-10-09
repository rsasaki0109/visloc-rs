//! A folder of photos -> trained splat -> mesh, in one command.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos -- \
//!     --images <folder> --out <work dir> [--steps 30000] [--max-size 1600] [--focal PX] \
//!     [--no-mesh] [--normal-weight 0.005] [--appearance] [--exhaustive-max 300] [--window 20]
//!     [--retrieval 30] [--max-keypoints 4000] [--sift-opt key=value ...] [--match-f32]
//! ```
//!
//! SfM from `visloc_gsplat_train::photos` (EXIF focal, GPU SIFT/matching,
//! the COLMAP-port mapper, intrinsics refinement), brush-strategy training,
//! then mesh extraction (`visloc_gsplat_train::mesh`). Writes
//! `<out>/scene.ply` and `<out>/mesh.ply`; no COLMAP or Python involved.

use std::path::PathBuf;
use std::time::Instant;

use visloc_gsplat_core::ply::save_ply;
use visloc_gsplat_render::GpuContext;
use visloc_gsplat_train::dataset::load_view_rgb;
use visloc_gsplat_train::init::seed_scene;
use visloc_gsplat_train::mesh::{extract_from_splat, MeshOptions};
use visloc_gsplat_train::metrics::psnr;
use visloc_gsplat_train::photos::{build_photo_dataset, PhotoSfmConfig};
use visloc_gsplat_train::trainer::{TrainConfig, Trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut images: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut sfm = PhotoSfmConfig::default();
    let mut steps = 30_000usize;
    let mut mesh = true;
    let mut normal_weight = 0.0f32;
    let mut appearance = false;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--images" => images = Some(PathBuf::from(next()?)),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--steps" => steps = next()?.parse()?,
            "--max-size" => sfm.max_size = next()?.parse()?,
            "--focal" => sfm.focal_px = Some(next()?.parse()?),
            "--no-refine-intrinsics" => sfm.refine_intrinsics = false,
            // Skip the final intrinsics refinement above this many
            // registered photos (default: never).
            "--refine-intrinsics-max" => sfm.refine_intrinsics_max_images = next()?.parse()?,
            // Match every pair up to this many photos; beyond it, a sliding
            // window of `--window` photos over the sorted file names.
            "--exhaustive-max" => sfm.exhaustive_max = next()?.parse()?,
            "--window" => sfm.window = next()?.parse()?,
            // Past --exhaustive-max: also match each photo with its N most
            // similar photos by appearance (VLAD), e.g. to join revisits.
            "--retrieval" => sfm.retrieval_k = next()?.parse()?,
            "--match-f32" => sfm.match_u8 = false,
            "--max-keypoints" => sfm.sift_max_keypoints = next()?.parse()?,
            // Extra SIFT settings, e.g. `--sift-opt affine=1 --sift-opt
            // domain_size_pooling=1` for strongly oblique / wide-baseline photos.
            "--sift-opt" => sfm.sift_overrides.push(next()?),
            "--cpu" => sfm.gpu = false,
            "--no-mesh" => mesh = false,
            "--normal-weight" => normal_weight = next()?.parse()?,
            // Per-photo colour correction (varying exposure / white balance).
            "--appearance" => appearance = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let images = images.ok_or("--images <folder> is required")?;
    let out = out.ok_or("--out <work dir> is required")?;
    std::fs::create_dir_all(&out)?;
    let t0 = Instant::now();

    let (dataset, points, report) =
        build_photo_dataset(&images, &out, &sfm, &mut |m: &str| println!("{m}"))?;
    println!(
        "sfm done in {:.1}s: {}/{} registered, focal {:.1} -> {:.1} px ({}), {} points",
        t0.elapsed().as_secs_f64(),
        report.registered,
        report.images,
        report.focal_prior,
        report.focal,
        report.focal_source,
        report.points
    );
    if steps == 0 {
        return Ok(());
    }

    let init = seed_scene(&points, 3);
    let cfg = TrainConfig {
        steps,
        normal_weight,
        appearance,
        ..TrainConfig::brush_preset()
    };
    let t_train = Instant::now();
    let mut trainer = Trainer::new(GpuContext::new()?, &dataset, &init, cfg)?;
    for step in 1..=steps {
        trainer.step()?;
        if step % 1000 == 0 {
            println!(
                "step {step:6}  l1 {:.4}  gaussians {}  ({:.0}s)",
                trainer.take_mean_loss(),
                trainer.num_gaussians(),
                t_train.elapsed().as_secs_f64()
            );
        }
    }
    let mut sum = 0.0;
    for v in &dataset.eval {
        sum += psnr(&trainer.render(&v.camera).rgb, &load_view_rgb(v)?);
    }
    println!(
        "trained {steps} steps in {:.1}s; held-out psnr {:.2} over {} views",
        t_train.elapsed().as_secs_f64(),
        sum / dataset.eval.len().max(1) as f64,
        dataset.eval.len()
    );
    let scene = trainer.scene();
    drop(trainer);
    let ply = out.join("scene.ply");
    save_ply(&scene, &ply)?;
    println!("wrote {}", ply.display());

    if mesh {
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
        let (m, summary, _) = extract_from_splat(
            GpuContext::new()?,
            &scene,
            &views,
            &support,
            &MeshOptions::default(),
        )?;
        println!("{summary}");
        let path = out.join("mesh.ply");
        m.write_ply(&path)?;
        println!("wrote {}", path.display());
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
