use serde_json::{json, Value};
use visloc_rs::localization::{
    align_sessions, register_map, MapMergeConfig, SessionAlignment, SessionAlignmentConfig,
    SessionConstraint,
};

use super::map_files::{self, SavedMap};

pub(super) fn solve(
    maps: &[SavedMap],
    config: &MapMergeConfig,
) -> Result<(SessionAlignment, Value), Box<dyn std::error::Error>> {
    let mut constraints = Vec::new();
    let mut pair_reports = Vec::new();
    let mut failed_pairs = Vec::new();
    for (base_id, base) in maps.iter().enumerate() {
        for (other_id, other) in maps.iter().enumerate().skip(base_id + 1) {
            match register_map(
                &base.map,
                &base.descriptors,
                &other.map,
                &other.descriptors,
                config,
            ) {
                Ok(registration) => {
                    println!(
                        "{base_id} <- {other_id}: {} agreeing / {} localized, {:.3} m RMS",
                        registration.agreeing_keyframes,
                        registration.localized_keyframes,
                        registration.keyframe_rms_m,
                    );
                    pair_reports.push(json!({
                        "base_session": base_id,
                        "other_session": other_id,
                        "base_from_other": map_files::se3_json(&registration.base_from_other),
                        "keyframes": registration.keyframes,
                        "localized_keyframes": registration.localized_keyframes,
                        "agreeing_keyframes": registration.agreeing_keyframes,
                        "keyframe_rms_m": registration.keyframe_rms_m,
                    }));
                    constraints.push(SessionConstraint {
                        base: base_id,
                        other: other_id,
                        base_from_other: registration.base_from_other,
                    });
                }
                Err(error) => {
                    println!("{base_id} <- {other_id}: {error}");
                    failed_pairs.push(json!({
                        "base_session": base_id,
                        "other_session": other_id,
                        "error": error.to_string(),
                    }));
                }
            }
        }
    }
    let alignment = align_sessions(
        maps.len(),
        &constraints,
        SessionAlignmentConfig {
            translation_scale_m: config.agreement_m,
            yaw_scale_rad: config.agreement_yaw_deg.to_radians(),
        },
    )?;
    for (pair, residual) in pair_reports.iter_mut().zip(&alignment.residuals) {
        pair["final_residual"] = json!({
            "translation_m": residual.translation_m,
            "yaw_deg": residual.yaw_rad.to_degrees(),
            "weight": residual.weight,
        });
    }

    println!(
        "optimized {} sessions, {} pair constraints; cost {:.6} -> {:.6} ({} iterations)",
        maps.len(),
        constraints.len(),
        alignment.cost_before,
        alignment.cost_after,
        alignment.iterations,
    );
    let report = json!({
        "anchor_session": 0,
        "session_count": maps.len(),
        "mode": "gravity_aligned_yaw_translation",
        "translation_scale_m": config.agreement_m,
        "yaw_scale_deg": config.agreement_yaw_deg,
        "cost_before": alignment.cost_before,
        "cost_after": alignment.cost_after,
        "iterations": alignment.iterations,
        "pairs": pair_reports,
        "failed_pairs": failed_pairs,
    });
    Ok((alignment, report))
}
