use std::fs;
use std::path::{Path, PathBuf};

use visloc_rs::vision::features::GrayscaleImage;
use visloc_rs::{
    EssentialMatrixLoopClosureVerifier, LoopClosureVerifierConfig, PairwiseLoopClosureScannerConfig,
};

use super::kitti_revisit_cli::CliArgs;
use super::kitti_revisit_frontend::FrontendReport;
use super::kitti_revisit_report_csv::write_candidates_csv;
use super::kitti_revisit_report_html::write_html_report;

pub(super) struct ReportInputs<'a> {
    pub(super) out_dir: &'a Path,
    pub(super) args: &'a CliArgs,
    pub(super) summary: &'a str,
    pub(super) reports: &'a [FrontendReport],
    pub(super) frame_paths: &'a [(u64, PathBuf)],
    pub(super) frame_dimensions: &'a [(u64, usize, usize)],
    pub(super) camera: &'a visloc_rs::Camera,
    pub(super) verifier: &'a EssentialMatrixLoopClosureVerifier,
    pub(super) segment_a_range: (u64, u64),
    pub(super) segment_b_range: (u64, u64),
    pub(super) scanner_cfg: &'a PairwiseLoopClosureScannerConfig,
    pub(super) verifier_cfg: &'a LoopClosureVerifierConfig,
}

pub(super) fn frame_dimensions(
    frame_ids: &[u64],
    images: &[&GrayscaleImage],
) -> Vec<(u64, usize, usize)> {
    frame_ids
        .iter()
        .zip(images.iter())
        .map(|(&id, image)| (id, image.width(), image.height()))
        .collect()
}

pub(super) fn write_report_bundle(
    inputs: ReportInputs<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(inputs.out_dir)?;
    fs::write(inputs.out_dir.join("summary.txt"), inputs.summary)?;
    write_candidates_csv(&inputs.out_dir.join("candidates.csv"), inputs.reports)?;
    write_html_report(&inputs)
}
