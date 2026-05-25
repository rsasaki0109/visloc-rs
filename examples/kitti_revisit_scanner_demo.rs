//! Real-data KITTI 00 long-revisit appearance-scanner demo.
//!
//! Loads two segments of KITTI odometry seq 00 (e.g. frames 0..49 from
//! the start and frames 4500..4529 from the major loop-closure point),
//! extracts FAST corners + intensity-patch descriptors per frame with
//! the project's `CornerFeatureExtractor`, and runs the appearance
//! `scan_pairwise_loop_closures` over all keyframes from both segments
//! at once. `min_keyframe_id_gap` is set high enough that only
//! cross-segment pairs are eligible; same-segment pairs (which would
//! always pair as "loops" since they sit one frame apart) are filtered
//! before the verifier ever fires.
//!
//! Validates that the scanner's defaults pick up real KITTI revisits
//! end-to-end: descriptor matching across an actual driving loop, not
//! just synthetic features. Companion to the synthetic
//! `scanner_loop_closure_demo`.
//!
//! Usage (after fetching both segments via
//! `scripts/fetch_kitti_seq00_images.py --start-frame N --max-frames K
//! --stride 1 --cameras image_0 --out-dir ...`):
//!
//!     cargo run --release --features image-io --example kitti_revisit_scanner_demo -- \
//!         --segment-a ~/datasets/kitti_seq00_stride1_subset/image_0 \
//!         --calib-a   ~/datasets/kitti_seq00_stride1_subset/calib.txt \
//!         --segment-b ~/datasets/kitti_seq00_revisit_4500/image_0 \
//!         --calib-b   ~/datasets/kitti_seq00_revisit_4500/calib.txt
//!
//! Both calibration files are typically the same KITTI 00 `calib.txt`;
//! the demo accepts independent paths so a future user can compare
//! across recalibrated splits.

// Submodules live in the `kitti_revisit/` subdirectory so Cargo does not try to
// build each one as a standalone example binary (only top-level `examples/*.rs`
// and `examples/*/main.rs` are auto-discovered). They stay direct children of
// this example crate root via `#[path]`, so their `super::` references resolve.
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_cli.rs"]
mod kitti_revisit_cli;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_config.rs"]
mod kitti_revisit_config;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_data.rs"]
mod kitti_revisit_data;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_format.rs"]
mod kitti_revisit_format;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_frontend.rs"]
mod kitti_revisit_frontend;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_overlay.rs"]
mod kitti_revisit_overlay;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_report.rs"]
mod kitti_revisit_report;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_report_csv.rs"]
mod kitti_revisit_report_csv;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_report_html.rs"]
mod kitti_revisit_report_html;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_report_html_parts.rs"]
mod kitti_revisit_report_html_parts;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_report_html_template.rs"]
mod kitti_revisit_report_html_template;
#[cfg(feature = "image-io")]
#[path = "kitti_revisit/kitti_revisit_summary.rs"]
mod kitti_revisit_summary;

#[cfg(feature = "image-io")]
use kitti_revisit_cli::parse_args;
#[cfg(feature = "image-io")]
use kitti_revisit_config::{build_scanner_config, build_verifier, print_scanner_config};
#[cfg(feature = "image-io")]
use kitti_revisit_data::load_revisit_dataset;
#[cfg(feature = "image-io")]
use kitti_revisit_frontend::run_selected_frontends;
#[cfg(feature = "image-io")]
use kitti_revisit_report::{frame_dimensions, write_report_bundle, ReportInputs};
#[cfg(feature = "image-io")]
use kitti_revisit_summary::build_summary;

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!("kitti_revisit_scanner_demo requires --features image-io");
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let dataset = load_revisit_dataset(&args)?;
    let images: Vec<_> = dataset.images.iter().collect();
    let scanner_cfg = build_scanner_config(&args, &dataset);
    print_scanner_config(&scanner_cfg);
    let verifier = build_verifier(&args);
    let reports = run_selected_frontends(&args, &dataset, &images, &scanner_cfg, &verifier)?;

    if let Some(out_dir) = &args.out_dir {
        let summary = build_summary(&args, &dataset, &scanner_cfg, &verifier.config, &reports);
        let dimensions = frame_dimensions(&dataset.frame_ids, &images);
        write_report_bundle(ReportInputs {
            out_dir,
            args: &args,
            summary: &summary,
            reports: &reports,
            frame_paths: &dataset.frame_paths,
            frame_dimensions: &dimensions,
            camera: &dataset.camera,
            verifier: &verifier,
            segment_a_range: dataset.segment_a_range,
            segment_b_range: dataset.segment_b_range,
            scanner_cfg: &scanner_cfg,
            verifier_cfg: &verifier.config,
        })?;
        println!("wrote {}/summary.txt", out_dir.display());
        println!("wrote {}/candidates.csv", out_dir.display());
        println!("wrote {}/index.html", out_dir.display());
    }
    Ok(())
}
