//! The relocalization attempt gates and their reject reasons.

use super::*;

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RelocalizationAttemptGateStatus {
    success_pass: bool,
    pub(super) min_inliers_pass: bool,
    pub(super) min_inlier_ratio_pass: bool,
    pub(super) reprojection_pass: bool,
    pub(super) continuity_pass: bool,
    pub(super) depth_ratio_pass: bool,
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RelocalizationAttemptGateConfig {
    pub(super) min_inliers: usize,
    pub(super) min_inlier_ratio: f64,
    pub(super) max_reprojection_error: Option<f64>,
    pub(super) max_translation_per_frame_from_last_success_meters: Option<f64>,
    pub(super) min_inlier_depth_median_ratio_to_last_success: Option<f64>,
    pub(super) max_inlier_depth_median_ratio_to_last_success: Option<f64>,
}

#[cfg(feature = "image-io")]
impl RelocalizationAttemptGateConfig {
    pub(super) fn from_args(args: &CliArgs) -> Self {
        Self {
            min_inliers: args.relocalization_min_inliers,
            min_inlier_ratio: args.relocalization_min_inlier_ratio,
            max_reprojection_error: args.relocalization_max_reprojection_error,
            max_translation_per_frame_from_last_success_meters: args
                .relocalization_max_translation_per_frame_from_last_success_meters,
            min_inlier_depth_median_ratio_to_last_success: args
                .relocalization_min_inlier_depth_median_ratio_to_last_success,
            max_inlier_depth_median_ratio_to_last_success: args
                .relocalization_max_inlier_depth_median_ratio_to_last_success,
        }
    }
}

#[cfg(feature = "image-io")]
pub(super) fn relocalization_attempt_gate_status(
    stats: &OnlineSlamRelocalizationStats,
    config: RelocalizationAttemptGateConfig,
) -> RelocalizationAttemptGateStatus {
    let reprojection_pass = match (config.max_reprojection_error, stats.mean_reprojection_error) {
        (Some(max), Some(actual)) => actual <= max,
        (Some(_), None) => false,
        (None, _) => true,
    };
    let continuity_pass = match (
        config.max_translation_per_frame_from_last_success_meters,
        stats.translation_per_frame_from_last_success_meters,
    ) {
        (Some(max), Some(actual)) => actual <= max,
        (Some(_), None) => true,
        (None, _) => true,
    };
    let depth_ratio_pass = match stats.inlier_depth_median_ratio_to_last_success {
        Some(ratio) => {
            config
                .min_inlier_depth_median_ratio_to_last_success
                .is_none_or(|min| ratio >= min)
                && config
                    .max_inlier_depth_median_ratio_to_last_success
                    .is_none_or(|max| ratio <= max)
        }
        None => true,
    };
    RelocalizationAttemptGateStatus {
        success_pass: stats.localization_success,
        min_inliers_pass: stats.inlier_count >= config.min_inliers,
        min_inlier_ratio_pass: stats.inlier_ratio >= config.min_inlier_ratio,
        reprojection_pass,
        continuity_pass,
        depth_ratio_pass,
    }
}

#[cfg(feature = "image-io")]
pub(super) fn relocalization_attempt_reject_reason(
    stats: &OnlineSlamRelocalizationStats,
    gates: RelocalizationAttemptGateStatus,
) -> &'static str {
    if stats.succeeded {
        "accepted"
    } else if !gates.success_pass {
        "no_pnp_solution"
    } else if !gates.min_inliers_pass {
        "min_inliers"
    } else if !gates.min_inlier_ratio_pass {
        "min_inlier_ratio"
    } else if !gates.reprojection_pass {
        "max_reprojection_error"
    } else if !gates.continuity_pass {
        "translation_per_frame_from_last_success"
    } else if !gates.depth_ratio_pass {
        "inlier_depth_median_ratio_to_last_success"
    } else if stats.passed_acceptance_gates
        && stats.confirmation_count < stats.confirmation_required_count
    {
        "confirmation_waiting"
    } else if stats.passed_acceptance_gates {
        "tracker_acceptance"
    } else {
        "unknown_gate"
    }
}
