use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::iter;

use serde_json::{json, Value};
use visloc_core::geometry::SE3;
use visloc_rs::localization::{inspect_map_registration, MapLoopCandidate};

use super::{map_files, Args};

pub(super) fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = iter::once(&args.base).chain(&args.add).collect();
    let mut input_paths = HashSet::new();
    let mut maps = Vec::with_capacity(paths.len());
    for path in &paths {
        if !input_paths.insert(path.canonicalize()?) {
            return Err(format!("{}: repeated input map", path.display()).into());
        }
        let map = map_files::load(path)?;
        if map.images.iter().any(|image| image.session != 0)
            || map.summary["sessions"]
                .as_array()
                .is_some_and(|sessions| sessions.len() > 1)
        {
            return Err(format!("{}: expected an original session map", path.display()).into());
        }
        if maps
            .first()
            .is_some_and(|base: &map_files::SavedMap| base.frontend != map.frontend)
        {
            return Err(format!(
                "{}: inconsistent feature extraction settings",
                path.display()
            )
            .into());
        }
        maps.push(map);
    }

    let mut pairs = Vec::new();
    let mut transforms = vec![vec![None; maps.len()]; maps.len()];
    for (base_id, base) in maps.iter().enumerate() {
        for (other_id, other) in maps.iter().enumerate().skip(base_id + 1) {
            let report = inspect_map_registration(
                &base.map,
                &base.descriptors,
                &other.map,
                &other.descriptors,
                &args.config,
            );
            let agreeing_centers = report
                .candidates
                .iter()
                .filter(|candidate| candidate.agrees_with_registration)
                .map(|candidate| &candidate.other_camera_center_m)
                .collect::<Vec<_>>();
            let mut span_squared = 0.0_f64;
            for (index, center) in agreeing_centers.iter().enumerate() {
                for other_center in agreeing_centers.iter().skip(index + 1) {
                    span_squared = span_squared.max((*center - *other_center).norm_squared());
                }
            }
            let span_m = span_squared.sqrt();
            let (registration, registration_error) = match report.registration {
                Ok(registration) => {
                    println!(
                        "{base_id} <- {other_id}: {} agreeing / {} localized, {:.3} m RMS, {:.2} m span",
                        registration.agreeing_keyframes,
                        registration.localized_keyframes,
                        registration.keyframe_rms_m,
                        span_m,
                    );
                    let value = json!({
                        "base_from_other": map_files::se3_json(&registration.base_from_other),
                        "keyframes": registration.keyframes,
                        "localized_keyframes": registration.localized_keyframes,
                        "agreeing_keyframes": registration.agreeing_keyframes,
                        "keyframe_rms_m": registration.keyframe_rms_m,
                        "agreeing_query_span_m": span_m,
                    });
                    transforms[base_id][other_id] = Some(registration.base_from_other);
                    (value, None)
                }
                Err(error) => {
                    println!("{base_id} <- {other_id}: {error}");
                    (Value::Null, Some(error.to_string()))
                }
            };
            pairs.push(json!({
                "base_session": base_id,
                "other_session": other_id,
                "registration": registration,
                "registration_error": registration_error,
                "candidates": report.candidates.iter().map(candidate_json).collect::<Vec<_>>(),
            }));
        }
    }

    let mut triangles = Vec::new();
    for (a, from_a) in transforms.iter().enumerate() {
        for (b, from_b) in transforms.iter().enumerate().skip(a + 1) {
            for (c, ac) in from_a.iter().enumerate().skip(b + 1) {
                let (Some(ab), Some(bc), Some(ac)) = (&from_a[b], &from_b[c], ac) else {
                    continue;
                };
                let (translation_m, rotation_deg) = triangle_error(ab, bc, ac);
                println!("triangle {a}-{b}-{c}: {translation_m:.3} m, {rotation_deg:.3} deg");
                triangles.push(json!({
                    "sessions": [a, b, c],
                    "translation_error_m": translation_m,
                    "rotation_error_deg": rotation_deg,
                }));
            }
        }
    }

    let config = &args.config;
    let sessions = paths
        .iter()
        .zip(&maps)
        .map(|(source, map)| {
            let mut images = map
                .images
                .iter()
                .filter(|image| image.cam_id == 0)
                .collect::<Vec<_>>();
            images.sort_by_key(|image| (image.timestamp_ns, image.frame_id, image.image_id));
            let centers = images
                .iter()
                .filter_map(|image| {
                    let pose = map
                        .map
                        .keyframes
                        .get(&image.image_id)?
                        .frame
                        .pose
                        .as_ref()?;
                    let p = pose.camera_center_world();
                    Some([p.x, p.y, p.z])
                })
                .collect::<Vec<_>>();
            json!({"source": source, "camera0_centers_m": centers})
        })
        .collect::<Vec<_>>();
    let output = json!({
        "schema": "visloc.basalt_loop_inspection.v1",
        "sessions": sessions,
        "registration_config": {
            "min_inliers": config.min_inliers,
            "min_keyframes": config.min_keyframes,
            "agreement_m": config.agreement_m,
            "agreement_yaw_deg": config.agreement_yaw_deg,
            "ransac_iterations": config.ransac_iterations,
            "reprojection_px": config.reprojection_px,
            "max_hamming": config.gates.max_distance,
            "descriptor_ratio": config.gates.ratio,
            "retrieval_keyframes": config.retrieval_keyframes,
            "retrieval_bits": config.retrieval_bits,
            "duplicate_radius_m": config.duplicate_radius_m,
        },
        "pairs": pairs,
        "triangles": triangles,
    });
    fs::create_dir_all(&args.out)?;
    let path = args.out.join("loop_inspection.json");
    let mut file = BufWriter::new(File::create(&path)?);
    serde_json::to_writer_pretty(&mut file, &output)?;
    writeln!(file)?;
    file.flush()?;
    println!("wrote {}", path.display());
    Ok(())
}

fn candidate_json(candidate: &MapLoopCandidate) -> Value {
    let center = candidate.other_camera_center_m;
    json!({
        "base_keyframe_id": candidate.base_keyframe_id,
        "other_keyframe_id": candidate.other_keyframe_id,
        "other_camera_from_base_camera": map_files::se3_json(&candidate.other_camera_from_base_camera),
        "inlier_count": candidate.inlier_count,
        "base_keyframe_inliers": candidate.base_keyframe_inliers,
        "mean_reprojection_error_px": candidate.mean_reprojection_error_px,
        "other_camera_center_m": [center.x, center.y, center.z],
        "agrees_with_registration": candidate.agrees_with_registration,
    })
}

fn triangle_error(ab: &SE3, bc: &SE3, ac: &SE3) -> (f64, f64) {
    let error = ac.inverse().compose(&ab.compose(bc));
    (
        error.translation.norm(),
        error.rotation.angle().to_degrees(),
    )
}

#[cfg(test)]
mod tests {
    use nalgebra::{UnitQuaternion, Vector3};

    use super::*;

    #[test]
    fn triangle_compares_the_direct_and_composed_transforms_in_the_same_frame() {
        let ab = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.5),
            Vector3::x(),
        );
        let bc = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, -0.2),
            Vector3::y(),
        );
        let ac = ab.compose(&bc);
        let (translation_m, rotation_deg) = triangle_error(&ab, &bc, &ac);
        assert!(translation_m < 1e-12);
        assert!(rotation_deg < 1e-12);
        assert!(triangle_error(&ab, &bc, &bc.compose(&ab)).0 > 0.1);
    }
}
