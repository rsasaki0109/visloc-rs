//! Dump the stages of the opt-in VLFeat-compatible SIFT detector.
//!
//! The output is intentionally metadata-only (no descriptors): one TSV per
//! image contains the extrema before orientation expansion, all oriented
//! copies, the per-locus orientation-capped set, and the final global cap.
//! This is an explicit audit tool and does not affect the normal extractor.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use visloc_rs::vision::features::sift::{
    diagnose_sift_vlfeat_detector, GrayImage, SiftConfig, SiftDetectionCandidate,
    SiftDetectorDiagnostics, SiftOrientedDetection,
};

fn usage() -> &'static str {
    "usage: vlfeat_detector_diagnostic <images-dir> <out-dir> [max-keypoints] [max-orientations] [bilinear]"
}

fn image_paths(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths: Vec<_> = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
                    )
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn write_candidate(out: &mut String, stage: &str, row: &SiftDetectionCandidate) {
    out.push_str(&format!(
        "{stage}\t{:.9}\t{:.9}\t{:.9}\tNaN\t{:.9}\t{}\t{}\t-1\t{:.9}\n",
        row.x, row.y, row.sigma, row.response, row.octave, row.level, row.edge_score,
    ));
}

fn write_oriented(out: &mut String, stage: &str, row: &SiftOrientedDetection) {
    out.push_str(&format!(
        "{stage}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}\t{}\tNaN\n",
        row.x,
        row.y,
        row.sigma,
        row.orientation,
        row.response,
        row.octave,
        row.level,
        row.orientation_index
    ));
}

fn write_diagnostics(
    path: &Path,
    diagnostics: &SiftDetectorDiagnostics,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from(
        "# stage\tx\ty\tsigma\torientation\tresponse\toctave\tlevel\torientation_index\tedge_score\n",
    );
    for row in &diagnostics.before_orientation {
        write_candidate(&mut text, "before_orientation", row);
    }
    for row in &diagnostics.after_orientation {
        write_oriented(&mut text, "after_orientation", row);
    }
    for row in &diagnostics.after_orientation_cap {
        write_oriented(&mut text, "after_orientation_cap", row);
    }
    for row in &diagnostics.after_cap {
        write_oriented(&mut text, "after_cap", row);
    }
    fs::write(path, text)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let images_dir = PathBuf::from(args.next().ok_or_else(usage)?);
    let output_dir = PathBuf::from(args.next().ok_or_else(usage)?);
    let max_keypoints = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8192);
    let max_keypoints = if max_keypoints == 0 {
        usize::MAX
    } else {
        max_keypoints
    };
    let max_orientations = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    let bilinear_orientations = match args.next().as_deref() {
        None => false,
        Some("bilinear") => true,
        Some(other) => return Err(format!("{}: unknown orientation mode {other}", usage()).into()),
    };
    if args.next().is_some() {
        return Err(usage().into());
    }

    fs::create_dir_all(&output_dir)?;
    let config = SiftConfig {
        max_keypoints,
        max_orientations,
        vlfeat_compatible_detector: true,
        vlfeat_bilinear_orientations: bilinear_orientations,
        ..SiftConfig::default()
    };
    let paths = image_paths(&images_dir)?;
    for (index, path) in paths.iter().enumerate() {
        let image = visloc_io::images::read_common_image(path)?;
        let gray = GrayImage::new(image.width(), image.height(), image.pixels())?;
        let diagnostics = diagnose_sift_vlfeat_detector(&gray, &config)?;
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("image");
        write_diagnostics(
            &output_dir.join(format!("{stem}_detector.tsv")),
            &diagnostics,
        )?;
        eprintln!(
            "[{}/{}] {stem}: candidates={} oriented={} orientation_cap={} global_cap={}",
            index + 1,
            paths.len(),
            diagnostics.before_orientation.len(),
            diagnostics.after_orientation.len(),
            diagnostics.after_orientation_cap.len(),
            diagnostics.after_cap.len()
        );
    }
    Ok(())
}
