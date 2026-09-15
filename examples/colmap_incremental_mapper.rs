//! CLI harness for the faithful COLMAP incremental-mapper port
//! (`visloc_slam::colmap_incremental`, milestone C2 of
//! `docs/colmap_rig_mapper_port_plan.md`).
//!
//! ```text
//! cargo run --release --example colmap_incremental_mapper -- \
//!   --manifest <rig manifest> --features-dir <dir> \
//!   --pairs-export <VISLOC-COLMAP-1 .bin> --out-colmap <dir> \
//!   [--random-seed 0] [--num-threads N] [--pose-solver gp3p|dlt6pt]
//!   [--local-ba-point-policy colmap|window|nopull]
//! ```
//!
//! Loads the same three inputs `DatabaseCache::from_generalized_rig_export`
//! consumes (rig manifest, per-image `*_features.txt` keypoints, a
//! COLMAP-frontend verified-pair export), runs
//! `colmap_incremental::pipeline::run`, and writes each kept sub-model to
//! `<out-colmap>/model/<k>/{cameras,images,points3D}.txt` (COLMAP text
//! format, via `Reconstruction::export_colmap_text`) plus a plain-text run
//! log (`<out-colmap>/mapper.log`) with per-registration lines (frame id,
//! path A/B, inlier counts, `num_reg_frames`) and BA events — sourced from
//! `pipeline::RunResult::log`, which is `IncrementalMapper::log` (this
//! port's `mapper.rs`) threaded out through `pipeline::run`.
//!
//! `--num-threads` is accepted for CLI-surface parity with COLMAP's own
//! `mapper` tool but is a no-op: this port's mapper is single-threaded (see
//! `mapper_impl.rs`'s module doc on `find_initial_image_pair`).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use visloc_rs::slam::colmap_incremental::{pipeline, DatabaseCache};

struct Args {
    manifest: PathBuf,
    features_dir: PathBuf,
    pairs_export: PathBuf,
    out_colmap: PathBuf,
    random_seed: u64,
    pose_solver: String,
    local_ba_point_policy: String,
}

fn parse_args() -> Args {
    let mut flags: HashMap<String, String> = HashMap::new();
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        let Some(name) = arg.strip_prefix("--") else {
            continue;
        };
        let value = argv.next().unwrap_or_default();
        flags.insert(name.to_string(), value);
    }

    let get = |key: &str| -> PathBuf {
        PathBuf::from(
            flags
                .get(key)
                .unwrap_or_else(|| panic!("missing required --{key}")),
        )
    };

    Args {
        manifest: get("manifest"),
        features_dir: get("features-dir"),
        pairs_export: get("pairs-export"),
        out_colmap: get("out-colmap"),
        random_seed: flags
            .get("random-seed")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0),
        pose_solver: flags
            .get("pose-solver")
            .cloned()
            .unwrap_or_else(|| "gp3p".to_string()),
        local_ba_point_policy: flags
            .get("local-ba-point-policy")
            .cloned()
            .unwrap_or_else(|| "colmap".to_string()),
    }
}

fn main() {
    let args = parse_args();
    let start = Instant::now();

    eprintln!(
        "loading database cache: manifest={} features_dir={} pairs_export={}",
        args.manifest.display(),
        args.features_dir.display(),
        args.pairs_export.display()
    );
    let db = DatabaseCache::from_generalized_rig_export(
        &args.manifest,
        &args.features_dir,
        &args.pairs_export,
    )
    .unwrap_or_else(|e| panic!("failed to load database cache: {e}"));
    eprintln!(
        "loaded: {} rigs, {} cameras, {} frames, {} images",
        db.num_rigs(),
        db.num_cameras(),
        db.num_frames(),
        db.num_images()
    );

    let mut options = pipeline::PipelineOptions::default();
    options.mapper.random_seed = args.random_seed;
    options.mapper.pose_solver = match args.pose_solver.as_str() {
        "gp3p" => visloc_slam::colmap_incremental::mapper::PoseSolverBackend::Gp3p,
        "dlt6pt" => visloc_slam::colmap_incremental::mapper::PoseSolverBackend::Dlt6pt,
        other => panic!("unknown --pose-solver {other} (gp3p|dlt6pt)"),
    };
    // `LocalBaPointPolicy` lives on `BundleAdjustmentOptions` (consumed
    // directly by `bundle_adjustment::solve`), not on `mapper::Options` —
    // `PipelineOptions::local_ba`/`global_ba` are independent
    // `BundleAdjustmentOptions` instances, so both are set here rather than
    // through `options.mapper`. `global_ba`'s copy is inert
    // (`adjust_global_bundle` never calls `add_variable_point`, see
    // `BundleAdjustmentOptions::global`'s doc) but is set for consistency.
    let local_ba_point_policy = match args.local_ba_point_policy.as_str() {
        "colmap" => visloc_slam::colmap_incremental::LocalBaPointPolicy::Colmap,
        "window" => visloc_slam::colmap_incremental::LocalBaPointPolicy::WindowOnly,
        "nopull" => visloc_slam::colmap_incremental::LocalBaPointPolicy::VariableWithoutPullIn,
        other => panic!("unknown --local-ba-point-policy {other} (colmap|window|nopull)"),
    };
    options.local_ba.local_ba_point_policy = local_ba_point_policy;
    options.global_ba.local_ba_point_policy = local_ba_point_policy;

    fs::create_dir_all(&args.out_colmap).expect("failed to create --out-colmap directory");

    let result = pipeline::run(&options, &db);
    let models = result.models;

    let elapsed = start.elapsed();
    eprintln!(
        "mapper finished in {:.1}s: {} model(s) kept",
        elapsed.as_secs_f64(),
        models.len()
    );

    let mut log_lines: Vec<String> = vec![
        format!(
            "colmap_incremental_mapper: manifest={} pairs_export={} random_seed={} pose_solver={} local_ba_point_policy={}",
            args.manifest.display(),
            args.pairs_export.display(),
            args.random_seed,
            args.pose_solver,
            args.local_ba_point_policy,
        ),
        format!(
            "database: {} rigs, {} cameras, {} frames, {} images",
            db.num_rigs(),
            db.num_cameras(),
            db.num_frames(),
            db.num_images()
        ),
    ];
    // Per-registration lines (frame id, path A/B, inliers, num_reg_frames)
    // and BA events, in chronological order, from `IncrementalMapper::log`.
    log_lines.extend(result.log);

    for (k, model) in models.iter().enumerate() {
        let recon = &model.reconstruction;
        let model_dir = args.out_colmap.join("model").join(k.to_string());
        let frame_count = recon
            .export_colmap_text(&model_dir)
            .unwrap_or_else(|e| panic!("failed to export model {k}: {e}"));
        eprintln!(
            "model {k}: {} registered images, {} points3D -> {}",
            frame_count,
            recon.num_points3d(),
            model_dir.display()
        );
        log_lines.push(format!(
            "MODEL {k}: registered_images={} points3D={} out={}",
            frame_count,
            recon.num_points3d(),
            model_dir.display()
        ));
    }
    log_lines.push(format!("wall_time_s={:.3}", elapsed.as_secs_f64()));

    let log_path = args.out_colmap.join("mapper.log");
    let mut full_log = log_lines.join("\n");
    full_log.push('\n');
    fs::write(&log_path, &full_log).expect("failed to write mapper.log");
    eprintln!("wrote log to {}", log_path.display());
}
