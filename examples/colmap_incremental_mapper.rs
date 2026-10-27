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
//!   [--init-image1 ID --init-image2 ID] [--gpu-ba 1]
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
//! `--gpu-ba 1` (or `--gpu-pcg 1`; a build with `--features gpu`) moves the
//! point elimination's off-diagonal blocks and the conjugate gradients of
//! the large bundle adjustments' reduced camera systems to the GPU
//! (`visloc_ba_gpu::GpuReducedSolver`); the result then differs from the CPU
//! solve at the f32 level instead of being bit-identical.
//!
//! `--num-threads` is accepted for CLI-surface parity with COLMAP's own
//! `mapper` tool but is a no-op: this port's mapper is single-threaded (see
//! `mapper_impl.rs`'s module doc on `find_initial_image_pair`).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use visloc_rs::slam::colmap_incremental::{pipeline, DatabaseCache};

// Opt-in global allocator swap (`--features mimalloc-global`): the mapper's
// bundle adjustments allocate and free gigabytes per call on large scenes.
// Pure runtime substitution -- does not change any algorithm's output. With
// `MIMALLOC_PURGE_DELAY=-1 MIMALLOC_ALLOW_LARGE_OS_PAGES=1` (keep freed
// memory, huge pages) a Courthouse replay took 36.2 s instead of 38.9 s
// with the system allocator (828k -> 715 page faults); mimalloc's default
// purging was slower than the system allocator (41.6 s).
#[cfg(feature = "mimalloc-global")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Args {
    manifest: PathBuf,
    features_dir: PathBuf,
    pairs_export: PathBuf,
    out_colmap: PathBuf,
    random_seed: u64,
    pose_solver: String,
    local_ba_point_policy: String,
    forced_init_pair: Option<(u64, u64)>,
    gpu_ba: bool,
}

const USAGE: &str = "\
usage: colmap_incremental_mapper --manifest <rig manifest> --features-dir <dir> \
--pairs-export <VISLOC-COLMAP-1 .bin> --out-colmap <dir> [options]

Runs the COLMAP-port incremental mapper on an export (gsplat_photos writes one
to <out>/port) and writes <out-colmap>/model/<k>/{cameras,images,points3D}.txt
and <out-colmap>/mapper.log. Mapper options recorded next to the manifest
(mapper_options.txt) are applied.

options:
  --random-seed 0  --pose-solver gp3p|dlt6pt
  --local-ba-point-policy colmap|window|nopull
  --init-image1 ID --init-image2 ID   force the initial pair
  --gpu-ba 1                          large bundle adjustments on the GPU";

/// Print `message` and the usage, then exit with status 2.
fn usage_error(message: &str) -> ! {
    eprintln!("{message}\n\n{USAGE}");
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut flags: HashMap<String, String> = HashMap::new();
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        if arg == "-h" || arg == "--help" {
            println!("{USAGE}");
            std::process::exit(0);
        }
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
                .unwrap_or_else(|| usage_error(&format!("missing required --{key}"))),
        )
    };

    let forced_init_pair = match (flags.get("init-image1"), flags.get("init-image2")) {
        (None, None) => None,
        (Some(first), Some(second)) => Some((
            first
                .parse::<u64>()
                .expect("--init-image1 must be an integer"),
            second
                .parse::<u64>()
                .expect("--init-image2 must be an integer"),
        )),
        _ => panic!("--init-image1 and --init-image2 must be given together"),
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
        forced_init_pair,
        gpu_ba: ["gpu-ba", "gpu-pcg"]
            .iter()
            .any(|k| flags.get(*k).is_some_and(|v| v != "0")),
    }
}

/// Registers the GPU reduced-camera solve; exits when it is unavailable, so
/// a GPU run never silently falls back to the CPU.
#[cfg(feature = "gpu")]
fn enable_gpu_ba() {
    match visloc_ba_gpu::register_reduced_pcg() {
        Ok(adapter) => eprintln!("gpu ba: reduced camera systems on {adapter}"),
        Err(e) => {
            eprintln!("gpu ba: no GPU ({e})");
            std::process::exit(2);
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn enable_gpu_ba() {
    eprintln!("gpu ba: this build lacks the `gpu` feature");
    std::process::exit(2);
}

/// `; rss=… MB peak=… MB` from `/proc/self/status` (Linux), else empty: the
/// database cache's share of a run's memory, next to the mapper's peak.
fn memory_note() -> String {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return String::new();
    };
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(String::from("?"), |kb| format!("{:.0}", kb / 1024.0))
    };
    format!("; rss={} MB peak={} MB", field("VmRSS:"), field("VmHWM:"))
}

fn main() {
    let args = parse_args();
    if args.gpu_ba {
        enable_gpu_ba();
    }
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
        "loaded: {} rigs, {} cameras, {} frames, {} images in {:.1} s{}",
        db.num_rigs(),
        db.num_cameras(),
        db.num_frames(),
        db.num_images(),
        start.elapsed().as_secs_f64(),
        memory_note()
    );

    let mut options = pipeline::PipelineOptions::default();
    options.mapper.random_seed = args.random_seed;
    // Exports from `gsplat_photos`/`gsplat_euroc` record the mapper options
    // they ran with (e.g. COLMAP's single-camera
    // `abs_pose_min_num_inliers=30`) next to the manifest; replay with them.
    let recorded = args
        .manifest
        .parent()
        .map(|dir| dir.join("mapper_options.txt"))
        .filter(|path| path.exists());
    if let Some(path) = recorded {
        let text = fs::read_to_string(&path).expect("failed to read mapper_options.txt");
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            options
                .apply_override(line)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        }
        eprintln!(
            "mapper options from {}: {}",
            path.display(),
            text.trim().replace('\n', ", ")
        );
    }
    // VISLOC_ABS_POSE_MIN_INLIERS overrides both.
    if let Some(n) = std::env::var("VISLOC_ABS_POSE_MIN_INLIERS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        options.mapper.abs_pose_min_num_inliers = n;
    }
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
    // Diagnostic (`plan` 4.2 item 1): force the initial pair to a chosen
    // pair of internal image ids; `None` keeps COLMAP's auto search.
    options.forced_init_pair = args.forced_init_pair;

    fs::create_dir_all(&args.out_colmap).expect("failed to create --out-colmap directory");

    let result = pipeline::run(&options, &db);
    eprintln!("mapper done{}", memory_note());
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

    let t_export = Instant::now();
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
    eprintln!("exported in {:.1} s", t_export.elapsed().as_secs_f64());
    log_lines.push(format!("wall_time_s={:.3}", elapsed.as_secs_f64()));

    let log_path = args.out_colmap.join("mapper.log");
    let mut full_log = log_lines.join("\n");
    full_log.push('\n');
    fs::write(&log_path, &full_log).expect("failed to write mapper.log");
    eprintln!("wrote log to {}", log_path.display());
}
