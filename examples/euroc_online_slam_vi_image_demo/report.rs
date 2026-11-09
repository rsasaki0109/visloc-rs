//! The ground-truth lookup and the log-line and CSV-row formatting.

use super::*;

#[cfg(feature = "image-io")]
pub(super) fn nearest_ground_truth(
    samples: &[EurocGroundTruthSample],
    target_ts: i128,
) -> Option<&EurocGroundTruthSample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples
        .binary_search_by_key(&target_ts, |sample| sample.timestamp_nanoseconds)
        .unwrap_or_else(|insert| {
            if insert == 0 {
                0
            } else if insert >= samples.len() {
                samples.len() - 1
            } else {
                let before = samples[insert - 1].timestamp_nanoseconds;
                let after = samples[insert].timestamp_nanoseconds;
                if (target_ts - before).abs() <= (after - target_ts).abs() {
                    insert - 1
                } else {
                    insert
                }
            }
        });
    samples.get(idx)
}

#[cfg(feature = "image-io")]
pub(super) fn format_vi_init_event(event: &ViInitializationEvent) -> String {
    match event {
        ViInitializationEvent::StillBuffering { reason } => {
            format!("StillBuffering reason={reason:?}")
        }
        ViInitializationEvent::Succeeded {
            result,
            first_keyframe_id,
            discarded_stale_factor_count,
        } => format!(
            "Succeeded first_keyframe={first_keyframe_id:?} discarded_stale={discarded_stale_factor_count} bias_gyro={:?} bias_acc={:?} rotation_angle_deg={:.4}",
            result.bias_gyro.as_slice(),
            result.bias_acc.as_slice(),
            result
                .initial_rotation_body_to_world
                .angle()
                .to_degrees(),
        ),
        ViInitializationEvent::GaveUp {
            last_reason,
            fallback,
        } => format!(
            "GaveUp last_reason={last_reason:?} fallback={fallback:?}",
        ),
    }
}

/// `last_gravity_alignment` is the initializer's most recent
/// `estimate_gravity_and_velocities` attempt (see
/// `MotionBasedViInitializer::last_gravity_alignment`), sampled at the same
/// frame as `event`. It is `None` when `--motion-vi-init-estimate-gravity`
/// is off. Threading it through separately from `event` is deliberate: on
/// real data most attempts are rejected by a downstream residual gate
/// (`ImuRawResidualOutOfRange` etc.), which carries no gravity vector of its
/// own, so without this the recovered estimate would be invisible on every
/// `StillWaiting` line — see `docs/motion_based_vi_alignment.md`'s
/// "Gravity-direction recovery" section.
/// `last_gyro_bias_alignment` is the initializer's most recent
/// `estimate_gyro_bias` attempt (see
/// `MotionBasedViInitializer::last_gyro_bias_alignment`), sampled at the
/// same frame as `event`, threaded through for the same reason as
/// `last_gravity_alignment` (see that parameter's doc comment): most
/// real-data attempts are rejected by a downstream gate that carries no
/// gyro-bias vector of its own.
#[cfg(feature = "image-io")]
pub(super) fn format_motion_vi_init_event(
    event: &MotionViInitializationEvent,
    last_gravity_alignment: Option<&GravityVelocityAlignment>,
    last_gyro_bias_alignment: Option<&GyroBiasAlignment>,
) -> String {
    match event {
        MotionViInitializationEvent::StillWaiting { reason } => {
            let alignment = last_gravity_alignment
                .map(|a| {
                    format!(
                        " gravity_alignment_gravity_world=[{:.6},{:.6},{:.6}] gravity_alignment_raw_norm={:.6} gravity_alignment_mean_residual_after={:.6} gravity_alignment_window_keyframes={}",
                        a.gravity_world.x,
                        a.gravity_world.y,
                        a.gravity_world.z,
                        a.raw_gravity_norm,
                        a.mean_residual_after,
                        a.window_keyframes,
                    )
                })
                .unwrap_or_default();
            let gyro_bias_alignment = last_gyro_bias_alignment
                .map(|a| {
                    format!(
                        " gyro_bias_alignment_bias_gyro=[{:.6},{:.6},{:.6}] gyro_bias_alignment_iterations={} gyro_bias_alignment_rms_before={:.6} gyro_bias_alignment_rms_after={:.6}",
                        a.bias_gyro.x,
                        a.bias_gyro.y,
                        a.bias_gyro.z,
                        a.iterations,
                        a.rotation_residual_rms_before,
                        a.rotation_residual_rms_after,
                    )
                })
                .unwrap_or_default();
            format!("StillWaiting reason={reason:?}{alignment}{gyro_bias_alignment}")
        }
        MotionViInitializationEvent::Succeeded { result } => format!(
            "Succeeded keyframes={} imu_factors={} scale={:.6} viba2_iters={} trigger_translation_m={:.3} bias_released={} estimated_gravity_world={:?} estimated_gyro_bias={:?}",
            result.keyframe_ids.len(),
            result.imu_factors_used,
            result.scale,
            result.viba2_iterations_run,
            result.trigger_translation_meters,
            result.bias_released,
            result.estimated_gravity_world.as_ref().map(|g| g.as_slice().to_vec()),
            result.estimated_gyro_bias.as_ref().map(|b| b.as_slice().to_vec()),
        ),
    }
}

#[cfg(feature = "image-io")]
pub(super) fn format_optional_frame_id(frame_id: Option<u64>) -> String {
    frame_id.map(|id| id.to_string()).unwrap_or_default()
}

#[cfg(feature = "image-io")]
pub(super) fn format_optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

#[cfg(feature = "image-io")]
pub(super) fn format_optional_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.9}")).unwrap_or_default()
}

#[cfg(feature = "image-io")]
pub(super) fn bool_as_u8(value: bool) -> u8 {
    if value {
        1
    } else {
        0
    }
}

#[cfg(feature = "image-io")]
pub(super) fn quote_csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(feature = "image-io")]
pub(super) fn append_atlas_alignment_attempt(
    csv: &mut String,
    attempt_idx: usize,
    frame_idx: usize,
    timestamp_ns: i128,
    alignment: &CrossSubmapAlignmentResult,
) {
    if alignment.diagnostics.is_empty() {
        csv.push_str(&format!(
            "{attempt_idx},{frame_idx},{timestamp_ns},{},{},,,,,,,,,,,,,,,0,no_ranked_candidate\n",
            alignment.source_submap_id, alignment.source_frame_id,
        ));
        return;
    }
    for diagnostic in &alignment.diagnostics {
        let scale = diagnostic.scale_estimate;
        let failure = diagnostic
            .failure_reason
            .as_ref()
            .map(|reason| quote_csv_field(&format!("{reason:?}")))
            .unwrap_or_default();
        let verified = alignment.verified_merge.as_ref().is_some_and(|merge| {
            merge.evidence.target_submap_id == diagnostic.target_submap_id
                && diagnostic.failure_reason.is_none()
        });
        let weldable_landmark_matches = alignment
            .verified_merge
            .as_ref()
            .filter(|merge| {
                merge.evidence.target_submap_id == diagnostic.target_submap_id
                    && diagnostic.failure_reason.is_none()
            })
            .map(|merge| merge.landmark_matches.len().to_string())
            .unwrap_or_default();
        csv.push_str(&format!(
            "{attempt_idx},{frame_idx},{timestamp_ns},{},{},{},{},{:.6},{},{},{},{},{:.6},{},{},{},{},{},{},{},{}\n",
            alignment.source_submap_id,
            alignment.source_frame_id,
            diagnostic.target_submap_id,
            diagnostic.target_frame_id,
            diagnostic.appearance_similarity,
            diagnostic.candidate_landmark_count,
            u8::from(diagnostic.used_projection_prior),
            diagnostic.localization_correspondence_count,
            diagnostic.localization_inlier_count,
            diagnostic.localization_inlier_ratio,
            diagnostic
                .mean_reprojection_error_px
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default(),
            scale
                .map(|estimate| format!("{:.9}", estimate.estimated_scale))
                .unwrap_or_default(),
            scale
                .map(|estimate| estimate.pair_count.to_string())
                .unwrap_or_default(),
            scale
                .map(|estimate| format!("{:.6}", estimate.inlier_ratio))
                .unwrap_or_default(),
            scale
                .map(|estimate| format!("{:.9}", estimate.median_absolute_deviation))
                .unwrap_or_default(),
            weldable_landmark_matches,
            u8::from(verified),
            failure,
        ));
    }
}

#[cfg(feature = "image-io")]
pub(super) fn append_atlas_boundary_factor(
    csv: &mut String,
    frame_idx: usize,
    timestamp_ns: i128,
    factor: &CrossSubmapBoundaryFactorResult,
) {
    let failure = factor
        .failure_reason
        .as_ref()
        .map(|reason| quote_csv_field(&format!("{reason:?}")))
        .unwrap_or_default();
    let scale = factor.scale_estimate;
    csv.push_str(&format!(
        "{frame_idx},{timestamp_ns},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{}\n",
        factor.source_submap_id,
        factor.source_frame_id,
        factor.target_submap_id,
        factor.target_frame_id,
        factor.descriptor_match_count,
        factor.spatial_mutual_match_count,
        factor.metric_correspondence_count,
        factor.projection_refined_correspondence_count,
        factor.projection_refinement_iterations,
        bool_as_u8(factor.used_transform_prior),
        factor.rigid_inlier_count,
        factor.rigid_inlier_ratio,
        factor
            .mean_target_reprojection_error_px
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
        scale
            .map(|estimate| format!("{:.9}", estimate.estimated_scale))
            .unwrap_or_default(),
        scale
            .map(|estimate| estimate.pair_count.to_string())
            .unwrap_or_default(),
        scale
            .map(|estimate| format!("{:.6}", estimate.inlier_ratio))
            .unwrap_or_default(),
        factor
            .verified_merge
            .as_ref()
            .map(|merge| merge.landmark_matches.len().to_string())
            .unwrap_or_default(),
        bool_as_u8(factor.verified_merge.is_some()),
        failure,
    ));
}

#[cfg(feature = "image-io")]
pub(super) fn keyframe_reason_csv_fields(
    reason: &KeyframeDecisionReason,
) -> (&'static str, String, String, String, String, String, String) {
    match reason {
        KeyframeDecisionReason::NotLocalized => (
            "NotLocalized",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::MissingPose => (
            "MissingPose",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::FirstSuccessfulFrame => (
            "FirstSuccessfulFrame",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::Relocalized => (
            "Relocalized",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::TrackedLandmarkDrop {
            frame_id_gap,
            tracked_landmarks,
            last_keyframe_tracked_landmarks,
            min_tracked_landmark_ratio,
        } => (
            "TrackedLandmarkDrop",
            frame_id_gap.to_string(),
            String::new(),
            String::new(),
            tracked_landmarks.to_string(),
            last_keyframe_tracked_landmarks.to_string(),
            format!("{min_tracked_landmark_ratio:.6}"),
        ),
        KeyframeDecisionReason::ThresholdsMet {
            frame_id_gap,
            translation,
        } => (
            "ThresholdsMet",
            frame_id_gap.to_string(),
            format!("{translation:.6}"),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::FrameIdGapTooSmall {
            frame_id_gap,
            min_frame_id_gap: _,
        } => (
            "FrameIdGapTooSmall",
            frame_id_gap.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::MaximumFrameIdGap {
            frame_id_gap,
            max_frame_id_gap: _,
        } => (
            "MaximumFrameIdGap",
            frame_id_gap.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::TranslationTooSmall {
            translation,
            min_translation,
        } => (
            "TranslationTooSmall",
            String::new(),
            format!("{translation:.6}"),
            format!("{min_translation:.6}"),
            String::new(),
            String::new(),
            String::new(),
        ),
        KeyframeDecisionReason::InsufficientTrackingQuality { .. } => (
            "InsufficientTrackingQuality",
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ),
    }
}
