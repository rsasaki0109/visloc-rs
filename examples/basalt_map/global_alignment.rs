use serde_json::{json, Value};
use visloc_rs::localization::{
    align_sessions, register_map, MapMergeConfig, MapMergeError, MapRegistration, SessionAlignment,
    SessionAlignmentConfig, SessionConstraint,
};

use super::map_files::{self, SavedMap};

pub(super) fn solve(
    maps: &mut [SavedMap],
    config: &MapMergeConfig,
    keyframe_loop_min_separation_m: Option<f64>,
    excluded_keyframe_loops: &std::collections::BTreeSet<[u64; 4]>,
) -> Result<(SessionAlignment, Value), Box<dyn std::error::Error>> {
    let mut constraints = Vec::new();
    let mut pair_reports = Vec::new();
    let mut directions = Vec::new();
    let mut failed_pairs = Vec::new();
    let register = |base_id: usize, other_id: usize| {
        let base = &maps[base_id];
        let other = &maps[other_id];
        register_map(
            &base.map,
            &base.descriptors,
            &other.map,
            &other.descriptors,
            config,
        )
    };
    for base_id in 0..maps.len() {
        for other_id in base_id + 1..maps.len() {
            match register_pair(base_id, other_id, register) {
                Ok((registration, forward_error)) => {
                    let (measurement_base, measurement_query) = if let Some(error) = &forward_error
                    {
                        println!(
                            "{base_id} <- {other_id}: {error}. Accepted reverse registration."
                        );
                        (other_id, base_id)
                    } else {
                        (base_id, other_id)
                    };
                    println!(
                        "{measurement_base} <- {measurement_query}: {} agreeing / {} localized, {:.3} m RMS",
                        registration.agreeing_keyframes,
                        registration.localized_keyframes,
                        registration.keyframe_rms_m,
                    );
                    pair_reports.push(json!({
                        "base_session": base_id,
                        "other_session": other_id,
                        "base_from_other": map_files::se3_json(&registration.base_from_other),
                        "measurement_base_session": measurement_base,
                        "measurement_query_session": measurement_query,
                        "forward_error": forward_error.map(|error| error.to_string()),
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
                    directions.push((measurement_base, measurement_query));
                }
                Err((error, reverse_error)) => {
                    println!("{base_id} <- {other_id}: {error}");
                    println!("{other_id} <- {base_id}: {reverse_error}");
                    failed_pairs.push(json!({
                        "base_session": base_id,
                        "other_session": other_id,
                        "error": error.to_string(),
                        "reverse_error": reverse_error.to_string(),
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
    let mut report = json!({
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
    if let Some(min_separation_m) = keyframe_loop_min_separation_m {
        report["keyframe_refinement"] = super::keyframe_alignment::refine(
            maps,
            &alignment.reference_from_session,
            &constraints,
            &directions,
            config,
            min_separation_m,
            excluded_keyframe_loops,
        )?;
    }
    Ok((alignment, report))
}

fn register_pair(
    base: usize,
    other: usize,
    mut register: impl FnMut(usize, usize) -> Result<MapRegistration, MapMergeError>,
) -> Result<(MapRegistration, Option<MapMergeError>), (MapMergeError, MapMergeError)> {
    let forward_error = match register(base, other) {
        Ok(registration) => return Ok((registration, None)),
        Err(error) => error,
    };
    match register(other, base) {
        Ok(mut registration) => {
            registration.base_from_other = registration.base_from_other.inverse();
            Ok((registration, Some(forward_error)))
        }
        Err(reverse_error) => Err((forward_error, reverse_error)),
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;

    use super::*;

    #[test]
    fn retries_failed_registration_and_inverts_reverse_measurement() {
        let reverse = MapRegistration {
            base_from_other: SE3::new(
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.4),
                Vector3::new(1.0, 2.0, 3.0),
            ),
            keyframes: 40,
            localized_keyframes: 30,
            agreeing_keyframes: 24,
            keyframe_rms_m: 0.02,
        };
        let mut attempts = Vec::new();
        let (registration, forward_error) = register_pair(1, 3, |base, other| {
            attempts.push((base, other));
            match (base, other) {
                (1, 3) => Err(MapMergeError::EmptyBaseMap),
                (3, 1) => Ok(reverse.clone()),
                _ => panic!("unexpected registration pair"),
            }
        })
        .unwrap();
        assert_eq!(attempts, [(1, 3), (3, 1)]);
        assert_eq!(forward_error, Some(MapMergeError::EmptyBaseMap));
        assert_eq!(registration.agreeing_keyframes, 24);
        let identity = reverse
            .base_from_other
            .compose(&registration.base_from_other);
        assert!(identity.translation.norm() < 1e-12);
        assert!(identity.rotation.angle() < 1e-12);

        let (registration, forward_error) = register_pair(1, 3, |base, other| {
            assert_eq!((base, other), (1, 3));
            Ok(reverse.clone())
        })
        .unwrap();
        assert_eq!(registration, reverse);
        assert_eq!(forward_error, None);
    }
}
