use visloc_rs::{
    EssentialMatrixLoopClosureVerifier, LoopClosureVerifierConfig, PairwiseLoopClosureScannerConfig,
};

use super::kitti_revisit_cli::CliArgs;
use super::kitti_revisit_data::RevisitDataset;

pub(super) fn build_scanner_config(
    args: &CliArgs,
    dataset: &RevisitDataset,
) -> PairwiseLoopClosureScannerConfig {
    PairwiseLoopClosureScannerConfig {
        min_keyframe_id_gap: dataset.min_keyframe_id_gap,
        min_matches: args.min_matches,
    }
}

pub(super) fn print_scanner_config(scanner_cfg: &PairwiseLoopClosureScannerConfig) {
    println!(
        "scanner config: min_keyframe_id_gap={} min_matches={}",
        scanner_cfg.min_keyframe_id_gap, scanner_cfg.min_matches,
    );
}

pub(super) fn build_verifier(args: &CliArgs) -> EssentialMatrixLoopClosureVerifier {
    EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: args.min_inliers,
            min_inlier_ratio: args.min_inlier_ratio,
            max_mean_sampson_error: args.max_mean_sampson_error,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    }
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
    fn scanner_config_uses_dataset_gap_and_arg_threshold() {
        let args = minimal_args();
        let dataset = minimal_dataset();

        let scanner_cfg = build_scanner_config(&args, &dataset);

        assert_eq!(scanner_cfg.min_keyframe_id_gap, 50);
        assert_eq!(scanner_cfg.min_matches, 30);
    }

    #[test]
    fn verifier_config_uses_cli_thresholds() {
        let args = minimal_args();

        let verifier = build_verifier(&args);

        assert_eq!(verifier.config.min_inliers, 12);
        assert_eq!(verifier.config.min_inlier_ratio, 0.4);
        assert_eq!(verifier.config.max_mean_sampson_error, 0.005);
        assert_eq!(verifier.config.default_translation_scale, 1.0);
    }
}
