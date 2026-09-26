//! Raw EuRoC sequence -> trained 3DGS scene, entirely in Rust (no COLMAP, no
//! Python): undistort cam0, SIFT, incremental SfM (visloc-rs), then train.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_euroc -- \
//!     --euroc E:/datasets/euroc_mav/all11/V1_01_easy --work out_dir \
//!     [--stride 4] [--max-frames 200] [--steps 7000] [--export-steps 7000] [--gpu-sift] [--gpu-ba]
//!     [--keypoints 8000] [--window 10] [--skips 15,20,30,45,60,90,120]
//!     [--ba-iterations 20] [--min-motion-px 2] [--render-dir <dir>]
//! ```
//!
//! `--euroc` is the directory containing `mav0/`. Undistorted frames go to
//! `<work>/images`, the splat to `<work>/scene.ply`. `--render-dir` also
//! renders every registered camera after training (`render_<frame>.png`, in
//! frame order: a flythrough along the recovered path).

use std::path::PathBuf;
use std::time::Instant;

use visloc_gsplat_core::ply::save_ply;
use visloc_gsplat_render::GpuContext;
use visloc_gsplat_train::dataset::load_view_rgb;
use visloc_gsplat_train::euroc::{build_euroc_dataset, EurocSfmConfig};
use visloc_gsplat_train::init::seed_scene;
use visloc_gsplat_train::metrics::{psnr, ssim};
use visloc_gsplat_train::trainer::{TrainConfig, Trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut euroc: Option<PathBuf> = None;
    let mut work = PathBuf::from("gsplat_euroc");
    let mut sfm = EurocSfmConfig::default();
    let mut cfg = TrainConfig::default();
    let mut render_dir: Option<PathBuf> = None;
    let mut export_steps: Vec<usize> = Vec::new();
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--euroc" => euroc = Some(PathBuf::from(next()?)),
            "--work" => work = PathBuf::from(next()?),
            "--stride" => sfm.stride = next()?.parse()?,
            "--max-frames" => sfm.max_frames = next()?.parse()?,
            "--gpu-sift" => sfm.gpu_sift = true,
            "--gpu-ba" => sfm.gpu_ba = true,
            "--keypoints" => sfm.sift_max_keypoints = next()?.parse()?,
            "--window" => sfm.window = next()?.parse()?,
            "--ba-iterations" => sfm.ba_max_iterations = Some(next()?.parse()?),
            "--local-ba-rel-tol" => sfm.local_ba_relative_tolerance = Some(next()?.parse()?),
            "--sfm-opt" => sfm.sfm_overrides.push(next()?),
            "--keep-planar" => sfm.keep_planar = true,
            "--no-panoramic" => sfm.keep_planar_no_panoramic = true,
            "--register-gated" => sfm.register_gated = true,
            "--merge-models" => sfm.merge_models = true,
            "--sift-l1-root" => sfm.sift_l1_root = true,
            "--sift-opt" => sfm.sift_overrides.push(next()?),
            "--import-features" => sfm.import_features = Some(PathBuf::from(next()?)),
            "--mapper" => match next()?.as_str() {
                "colmap-port" => sfm.colmap_port_mapper = true,
                "incremental" => sfm.colmap_port_mapper = false,
                other => return Err(format!("--mapper {other}: colmap-port|incremental").into()),
            },
            "--init-poses" => sfm.init_poses = Some(PathBuf::from(next()?)),
            "--import-colmap" => sfm.import_colmap = Some(PathBuf::from(next()?)),
            "--verify-min-inliers" => sfm.verify_min_inliers = Some(next()?.parse()?),
            "--min-motion-px" => sfm.min_keyframe_motion_px = next()?.parse()?,
            "--skips" => {
                sfm.skip_offsets = next()?
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--steps" => cfg.steps = next()?.parse()?,
            "--render-dir" => render_dir = Some(PathBuf::from(next()?)),
            "--export-steps" => {
                export_steps = next()?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let euroc = euroc.ok_or("--euroc <sequence dir containing mav0/> is required")?;

    let t0 = Instant::now();
    let (dataset, points, report) = build_euroc_dataset(&euroc, &work, &sfm, &mut |m| {
        println!("[{:6.1}s] {m}", t0.elapsed().as_secs_f64())
    })?;
    println!(
        "sfm done in {:.1}s: {}/{} frames registered, {} points, {} pairs, {:.3} px, ATE {}",
        t0.elapsed().as_secs_f64(),
        report.registered,
        report.frames,
        report.points,
        report.pairs,
        report.mean_reprojection_px,
        report
            .ate_rmse_m
            .map_or("n/a".to_string(), |a| format!("{:.2} cm", a * 100.0))
    );
    let init = seed_scene(&points, 3);
    let eval_gt: Vec<Vec<[f32; 3]>> = dataset
        .eval
        .iter()
        .map(load_view_rgb)
        .collect::<Result<_, _>>()?;

    let ctx = GpuContext::new()?;
    let mut trainer = Trainer::new(ctx, &dataset, &init, cfg.clone())?;
    let train_start = Instant::now();
    let ply = work.join("scene.ply");
    for step in 1..=cfg.steps {
        trainer.step()?;
        if export_steps.contains(&step) {
            let p = work.join(format!("scene_{step:05}.ply"));
            save_ply(&trainer.scene(), &p)?;
        }
        if step % 1000 == 0 {
            let loss = trainer.take_mean_loss();
            println!(
                "step {step:6}  l1 {loss:.4}  gaussians {}  ({:.0}s)",
                trainer.num_gaussians(),
                train_start.elapsed().as_secs_f64()
            );
        }
    }
    let (mut p_sum, mut s_sum) = (0.0, 0.0);
    for (v, gt) in dataset.eval.iter().zip(&eval_gt) {
        let img = trainer.render(&v.camera);
        p_sum += psnr(&img.rgb, gt);
        s_sum += ssim(
            &img.rgb,
            gt,
            v.camera.camera.width as usize,
            v.camera.camera.height as usize,
        );
    }
    let n = dataset.eval.len().max(1) as f64;
    println!(
        "trained {} steps in {:.1}s; eval ({} views) psnr {:.3} ssim {:.4}; total {:.1}s",
        cfg.steps,
        train_start.elapsed().as_secs_f64(),
        dataset.eval.len(),
        p_sum / n,
        s_sum / n,
        t0.elapsed().as_secs_f64()
    );
    save_ply(&trainer.scene(), &ply)?;
    println!("wrote {}", ply.display());
    if let Some(dir) = render_dir {
        // Every registered view, in frame order: a flythrough along the
        // recovered camera path (render_<frame>.png next to the input frame).
        std::fs::create_dir_all(&dir)?;
        let mut views: Vec<_> = dataset.train.iter().chain(&dataset.eval).collect();
        views.sort_by(|a, b| a.name.cmp(&b.name));
        for v in &views {
            let img = trainer.render(&v.camera);
            let (w, h) = (v.camera.camera.width, v.camera.camera.height);
            let bytes: Vec<u8> = img
                .rgb
                .iter()
                .flat_map(|p| p.iter().map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8))
                .collect();
            let out = dir.join(format!("render_{}", v.name));
            image::save_buffer(&out, &bytes, w, h, image::ColorType::Rgb8)?;
        }
        println!("rendered {} views to {}", views.len(), dir.display());
    }
    Ok(())
}
