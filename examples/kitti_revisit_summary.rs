use visloc_rs::{
    LoopClosureCandidate, LoopClosureVerifierConfig, PairwiseLoopClosureScannerConfig,
};

use super::kitti_revisit_cli::CliArgs;
use super::kitti_revisit_data::RevisitDataset;
use super::kitti_revisit_frontend::FrontendReport;

pub(super) fn build_summary(
    args: &CliArgs,
    dataset: &RevisitDataset,
    scanner_cfg: &PairwiseLoopClosureScannerConfig,
    verifier_cfg: &LoopClosureVerifierConfig,
    reports: &[FrontendReport],
) -> String {
    let mut summary = String::new();
    summary.push_str(&run_header(args, dataset, scanner_cfg, verifier_cfg));
    for report in reports {
        summary.push_str(&frontend_summary(report));
    }
    summary
}

fn run_header(
    args: &CliArgs,
    dataset: &RevisitDataset,
    scanner_cfg: &PairwiseLoopClosureScannerConfig,
    verifier_cfg: &LoopClosureVerifierConfig,
) -> String {
    format!(
        "min_keyframe_id_gap={} segment_a_frames={} segment_b_frames={} max_features={} min_matches={} min_inliers={} min_inlier_ratio={:.4} max_mean_sampson_error={:.6}\n",
        scanner_cfg.min_keyframe_id_gap,
        dataset.segment_a_len,
        dataset.segment_b_len,
        args.max_features,
        scanner_cfg.min_matches,
        verifier_cfg.min_inliers,
        verifier_cfg.min_inlier_ratio,
        verifier_cfg.max_mean_sampson_error,
    )
}

fn frontend_summary(report: &FrontendReport) -> String {
    let mut summary = String::new();
    summary.push_str(&format!("[{}]\n", report.label));
    summary.push_str(&format!(
        "  feature_count_min={} feature_count_max={}\n",
        report.feature_min, report.feature_max
    ));
    summary.push_str(&format!(
        "  extract_ms={} scan_ms={} total_ms={}\n",
        report.elapsed_extract_ms, report.elapsed_scan_ms, report.elapsed_total_ms
    ));
    summary.push_str(&format!("  candidates={}\n", report.candidates.len()));
    for candidate in &report.candidates {
        summary.push_str(&candidate_summary(candidate));
    }
    if let Some(strongest) = report.strongest() {
        summary.push_str(&format!(
            "  strongest_from={} strongest_to={} strongest_score={:.4}\n",
            strongest.matched_keyframe_id, strongest.query_frame_id, strongest.score,
        ));
    } else {
        summary.push_str("  strongest=None\n");
    }
    summary.push('\n');
    summary
}

fn candidate_summary(candidate: &LoopClosureCandidate) -> String {
    let verification = candidate
        .verification
        .as_ref()
        .expect("verification populated");
    format!(
        "  pair from={} to={} inliers={} ratio={:.4} mean_sampson={:.5} score={:.4}\n",
        candidate.matched_keyframe_id,
        candidate.query_frame_id,
        verification.inlier_count,
        verification.inlier_ratio,
        verification.mean_sampson_error,
        candidate.score,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use super::super::kitti_revisit_cli::FrontendChoice;

    fn minimal_args() -> CliArgs {
        CliArgs {
            segment_a: PathBuf::from("a/image_0"),
            segment_b: PathBuf::from("b/image_0"),
            calib_a: PathBuf::from("a/calib.txt"),
            calib_b: PathBuf::from("b/calib.txt"),
            projection_label: "P0".to_string(),
            out_dir: None,
            frontend: FrontendChoice::Deep,
            max_features: 200,
            min_matches: 30,
            min_inliers: 12,
            min_inlier_ratio: 0.4,
            max_mean_sampson_error: 0.005,
        }
    }

    fn minimal_dataset() -> RevisitDataset {
        RevisitDataset {
            camera: visloc_rs::Camera::pinhole(0, 1241, 376, 718.856, 718.856, 607.1928, 185.2157),
            images: Vec::new(),
            frame_ids: Vec::new(),
            frame_paths: Vec::new(),
            segment_a_len: 50,
            segment_b_len: 30,
            segment_a_range: (0, 49),
            segment_b_range: (4500, 4529),
            min_keyframe_id_gap: 50,
        }
    }

    #[test]
    fn run_header_records_thresholds_and_dataset_shape() {
        let args = minimal_args();
        let dataset = minimal_dataset();
        let scanner_cfg = PairwiseLoopClosureScannerConfig {
            min_keyframe_id_gap: 50,
            min_matches: 30,
        };
        let verifier_cfg = LoopClosureVerifierConfig {
            min_inliers: 12,
            min_inlier_ratio: 0.4,
            max_mean_sampson_error: 0.005,
            default_translation_scale: 1.0,
        };

        let summary = build_summary(&args, &dataset, &scanner_cfg, &verifier_cfg, &[]);

        assert!(summary.contains("min_keyframe_id_gap=50"));
        assert!(summary.contains("segment_a_frames=50"));
        assert!(summary.contains("segment_b_frames=30"));
        assert!(summary.contains("max_features=200"));
        assert!(summary.contains("max_mean_sampson_error=0.005000"));
    }

    #[test]
    fn frontend_summary_records_empty_report() {
        let report = FrontendReport {
            choice: FrontendChoice::Deep,
            label: "deep-style (HogLike + MutualSoftmax)",
            feature_min: 0,
            feature_max: 0,
            features: Vec::new(),
            candidates: Vec::new(),
            elapsed_total_ms: 3,
            elapsed_extract_ms: 1,
            elapsed_scan_ms: 2,
        };

        let summary = frontend_summary(&report);

        assert!(summary.contains("[deep-style (HogLike + MutualSoftmax)]"));
        assert!(summary.contains("candidates=0"));
        assert!(summary.contains("strongest=None"));
    }
}
