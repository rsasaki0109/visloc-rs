//! Per-pair diagnose output, ground-truth bearing checks and view-graph component export.

use super::*;

/// The profiles shared by the human-readable `--diagnose-pair` output and the
/// machine-readable CSV export. Keeping this list in one place prevents the
/// two diagnostics from silently measuring different NN match sets.
pub(super) const DIAGNOSE_PROFILES: [(f32, bool); 4] =
    [(0.8, true), (0.9, true), (0.95, true), (0.95, false)];

struct DiagnosePairProfile {
    matches: Vec<DescriptorMatch>,
    valid_matches: Vec<(usize, usize)>,
    correspondences: Vec<TwoViewCorrespondence>,
    report: Option<TwoViewGeometryReport>,
}

fn diagnose_pair_profile(
    features: &[FeatureSet],
    camera: &Camera,
    verifier: &TwoViewGeometryVerifier,
    matcher: &PairMatcher,
    i: usize,
    j: usize,
    ratio: f32,
    cross_check: bool,
) -> DiagnosePairProfile {
    let matches = matcher.match_pair(ratio, cross_check, i, j, &features[i], &features[j]);
    let valid_matches: Vec<(usize, usize)> = matches
        .iter()
        .filter_map(|m| {
            features[i]
                .keypoints
                .get(m.query_index)
                .and_then(|_| features[j].keypoints.get(m.train_index))
                .map(|_| (m.query_index, m.train_index))
        })
        .collect();
    let correspondences: Vec<TwoViewCorrespondence> = valid_matches
        .iter()
        .map(|&(query_index, train_index)| {
            TwoViewCorrespondence::new(
                features[i].keypoints[query_index],
                features[j].keypoints[train_index],
            )
        })
        .collect();
    let report = (correspondences.len() >= 8).then(|| verifier.classify(&correspondences, camera));
    DiagnosePairProfile {
        matches,
        valid_matches,
        correspondences,
        report,
    }
}

struct DiagnoseImportedRaw {
    valid_matches: Vec<(usize, usize)>,
    report: Option<TwoViewGeometryReport>,
}

fn diagnose_imported_raw(
    features: &[FeatureSet],
    camera: &Camera,
    verifier: &TwoViewGeometryVerifier,
    i: usize,
    j: usize,
    raw_matches: &[(usize, usize)],
) -> DiagnoseImportedRaw {
    let valid_matches: Vec<(usize, usize)> = raw_matches
        .iter()
        .copied()
        .filter(|&(query_index, train_index)| {
            features[i].keypoints.get(query_index).is_some()
                && features[j].keypoints.get(train_index).is_some()
        })
        .collect();
    let correspondences: Vec<TwoViewCorrespondence> = valid_matches
        .iter()
        .map(|&(query_index, train_index)| {
            TwoViewCorrespondence::new(
                features[i].keypoints[query_index],
                features[j].keypoints[train_index],
            )
        })
        .collect();
    let report = (correspondences.len() >= 8).then(|| verifier.classify(&correspondences, camera));
    DiagnoseImportedRaw {
        valid_matches,
        report,
    }
}

#[derive(Debug, Clone, Copy)]
struct DiagnosePairRow {
    ratio: f32,
    cross_check: bool,
    raw_matches: usize,
    valid_matches: usize,
    config: ConfigurationType,
    accepted_inliers: usize,
    e_inliers: usize,
    f_inliers: usize,
    h_inliers: usize,
    colmap_pair_present: bool,
    colmap_raw_matches: usize,
    colmap_index_overlap: usize,
    colmap_verified_present: bool,
    colmap_verified_inliers: usize,
    colmap_verified_config: Option<ConfigurationType>,
    imported_config: Option<ConfigurationType>,
    imported_accepted_inliers: usize,
    imported_e_inliers: usize,
    imported_f_inliers: usize,
    imported_h_inliers: usize,
    imported_accepted_index_overlap: usize,
}

pub(super) const fn configuration_name(config: ConfigurationType) -> &'static str {
    match config {
        ConfigurationType::Undefined => "UNDEFINED",
        ConfigurationType::Degenerate => "DEGENERATE",
        ConfigurationType::Uncalibrated => "UNCALIBRATED",
        ConfigurationType::Calibrated => "CALIBRATED",
        ConfigurationType::Planar => "PLANAR",
        ConfigurationType::Panoramic => "PANORAMIC",
        ConfigurationType::PlanarOrPanoramic => "PLANAR_OR_PANORAMIC",
        ConfigurationType::Watermark => "WATERMARK",
        ConfigurationType::Multiple => "MULTIPLE",
    }
}

fn diagnose_pair_row(
    features: &[FeatureSet],
    camera: &Camera,
    verifier: &TwoViewGeometryVerifier,
    matcher: &PairMatcher,
    i: usize,
    j: usize,
    ratio: f32,
    cross_check: bool,
    colmap_matches: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    colmap_verified: Option<&HashMap<(usize, usize), VerifiedPairOracle>>,
    imported_raw: Option<&DiagnoseImportedRaw>,
) -> DiagnosePairRow {
    let profile = diagnose_pair_profile(
        features,
        camera,
        verifier,
        matcher,
        i,
        j,
        ratio,
        cross_check,
    );
    let (config, accepted_inliers, e_inliers, f_inliers, h_inliers) = profile
        .report
        .as_ref()
        .map_or((ConfigurationType::Undefined, 0, 0, 0, 0), |report| {
            (
                report.config,
                report.inliers.len(),
                report.e_inlier_count,
                report.f_inlier_count,
                report.h_inlier_count,
            )
        });

    let colmap = colmap_matches.and_then(|matches| matches.get(&(i, j)));
    let colmap_set: HashSet<(usize, usize)> = colmap
        .into_iter()
        .flat_map(|matches| matches.iter().copied())
        .collect();
    let visloc_set: HashSet<(usize, usize)> = profile
        .matches
        .iter()
        .map(|m| (m.query_index, m.train_index))
        .collect();
    let colmap_index_overlap = visloc_set.intersection(&colmap_set).count();
    let verified = colmap_verified.and_then(|matches| matches.get(&(i, j)));
    let (
        imported_config,
        imported_accepted_inliers,
        imported_e_inliers,
        imported_f_inliers,
        imported_h_inliers,
    ) = imported_raw
        .and_then(|raw| raw.report.as_ref())
        .map_or((None, 0, 0, 0, 0), |report| {
            (
                Some(report.config),
                report.inliers.len(),
                report.e_inlier_count,
                report.f_inlier_count,
                report.h_inlier_count,
            )
        });
    let profile_accepted_set: HashSet<(usize, usize)> = profile
        .report
        .as_ref()
        .map(|report| {
            report
                .inliers
                .iter()
                .filter_map(|&idx| profile.valid_matches.get(idx).copied())
                .collect()
        })
        .unwrap_or_default();
    let imported_accepted_set: HashSet<(usize, usize)> = imported_raw
        .map(|raw| {
            raw.report
                .as_ref()
                .map(|report| {
                    report
                        .inliers
                        .iter()
                        .filter_map(|&idx| raw.valid_matches.get(idx).copied())
                        .collect()
                })
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let imported_accepted_index_overlap = profile_accepted_set
        .intersection(&imported_accepted_set)
        .count();

    DiagnosePairRow {
        ratio,
        cross_check,
        raw_matches: profile.matches.len(),
        valid_matches: profile.correspondences.len(),
        config,
        accepted_inliers,
        e_inliers,
        f_inliers,
        h_inliers,
        colmap_pair_present: colmap.is_some(),
        colmap_raw_matches: colmap.map_or(0, |matches| matches.len()),
        colmap_index_overlap,
        colmap_verified_present: verified.is_some(),
        colmap_verified_inliers: verified.map_or(0, |oracle| oracle.inliers),
        colmap_verified_config: verified.map(|oracle| oracle.config),
        imported_config,
        imported_accepted_inliers,
        imported_e_inliers,
        imported_f_inliers,
        imported_h_inliers,
        imported_accepted_index_overlap,
    }
}

pub(super) fn diagnose_pairs_for_csv(
    features: &[FeatureSet],
    image_names: &[String],
    args: &Args,
) -> Result<Vec<(usize, usize)>, String> {
    if args.diagnose_pair_stems.is_empty() {
        let generated = if let Some(path) = args.candidate_manifest.as_deref() {
            parse_candidate_manifest(path, image_names)?
        } else {
            candidate_pairs(features, image_names, args)?
        };
        return filter_pairs_by_stem_window(generated, image_names, args.pair_stem_window);
    }
    filter_pairs_by_stem_window(
        all_pairs(features.len())
            .into_iter()
            .filter(|&(i, j)| {
                args.diagnose_pair_stems.iter().any(|stem| {
                    image_stem(&image_names[i]) == stem || image_stem(&image_names[j]) == stem
                })
            })
            .collect(),
        image_names,
        args.pair_stem_window,
    )
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

pub(super) fn write_diagnose_pairs_csv(
    path: &Path,
    features: &[FeatureSet],
    image_names: &[String],
    pairs: &[(usize, usize)],
    camera: &Camera,
    matcher: &PairMatcher,
    colmap_matches: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    colmap_verified: Option<&HashMap<(usize, usize), VerifiedPairOracle>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        writer,
        "image_i,image_j,image_name_i,image_name_j,kp_i,kp_j,ratio,cross_check,raw_matches,valid_matches,config,accepted_inliers,e_inliers,f_inliers,h_inliers,colmap_pair_present,colmap_raw_matches,colmap_index_overlap,colmap_verified_present,colmap_verified_inliers,colmap_verified_config,imported_config,imported_accepted_inliers,imported_e_inliers,imported_f_inliers,imported_h_inliers,imported_accepted_index_overlap"
    )?;
    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));
    let mut rows = 0usize;
    for &(i, j) in pairs {
        let imported_raw = colmap_matches
            .and_then(|matches| matches.get(&(i, j)))
            .map(|matches| diagnose_imported_raw(features, camera, &verifier, i, j, matches));
        for &(ratio, cross_check) in &DIAGNOSE_PROFILES {
            let row = diagnose_pair_row(
                features,
                camera,
                &verifier,
                matcher,
                i,
                j,
                ratio,
                cross_check,
                colmap_matches,
                colmap_verified,
                imported_raw.as_ref(),
            );
            writeln!(
                writer,
                "{i},{j},{},{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                csv_escape(&image_names[i]),
                csv_escape(&image_names[j]),
                features[i].keypoints.len(),
                features[j].keypoints.len(),
                row.ratio,
                if row.cross_check { 1 } else { 0 },
                row.raw_matches,
                row.valid_matches,
                configuration_name(row.config),
                row.accepted_inliers,
                row.e_inliers,
                row.f_inliers,
                row.h_inliers,
                if row.colmap_pair_present { 1 } else { 0 },
                row.colmap_raw_matches,
                row.colmap_index_overlap,
                if row.colmap_verified_present { 1 } else { 0 },
                row.colmap_verified_inliers,
                row.colmap_verified_config
                    .map_or("NONE", configuration_name),
                row.imported_config
                    .map_or("NONE", configuration_name),
                row.imported_accepted_inliers,
                row.imported_e_inliers,
                row.imported_f_inliers,
                row.imported_h_inliers,
                row.imported_accepted_index_overlap,
            )?;
            rows += 1;
        }
    }
    writer.flush()?;
    Ok(rows)
}

/// M5 diagnosis tool (`--diagnose-pair I,J`): dump raw match counts and
/// [`TwoViewGeometryVerifier`] outcomes for one specific `(i, j)` image pair
/// across the same fixed battery used by `--diagnose-pairs-csv`.
pub(super) fn diagnose_pair(
    features: &[FeatureSet],
    camera: &Camera,
    matcher: &PairMatcher,
    i: usize,
    j: usize,
) {
    println!("=== diagnose-pair ({i}, {j}) ===");
    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));
    for &(ratio, cross_check) in &DIAGNOSE_PROFILES {
        let profile = diagnose_pair_profile(
            features,
            camera,
            &verifier,
            matcher,
            i,
            j,
            ratio,
            cross_check,
        );
        let Some(report) = profile.report.as_ref() else {
            let detail = if profile.matches.len() < 8 {
                "too few to classify"
            } else {
                "invalid keypoint index"
            };
            println!(
                "  ratio={ratio:.2} cross_check={cross_check:<5} raw_matches={:<5} ({detail})",
                profile.matches.len()
            );
            continue;
        };
        // Recover a translation direction from E inliers when present, for
        // façade chirality diagnosis against GT.
        let mut pose_note = String::new();
        if report.e_inlier_count >= 8 {
            let e_corrs: Vec<TwoViewCorrespondence> = report
                .essential_inliers
                .iter()
                .filter_map(|&idx| profile.correspondences.get(idx).copied())
                .collect();
            if let Some(rel) = RelativePoseEstimator::default().estimate(&e_corrs, camera) {
                let t = rel.previous_to_current.translation;
                let r = rel.previous_to_current.rotation;
                if let Some(d) = (-r.inverse().transform_vector(&t)).try_normalize(1e-12) {
                    pose_note = format!(" E_dir=[{:.3},{:.3},{:.3}]", d.x, d.y, d.z);
                }
            }
        }
        println!(
            "  ratio={ratio:.2} cross_check={cross_check:<5} raw_matches={:<5} config={:?} \
             inliers={} (E={} F={} H={}){pose_note}",
            profile.matches.len(),
            report.config,
            report.inliers.len(),
            report.e_inlier_count,
            report.f_inlier_count,
            report.h_inlier_count,
        );
    }
}

pub(super) fn gt_poses_aligned(
    image_names: &[String],
    by_stem: &HashMap<String, Pose>,
) -> Vec<Option<Pose>> {
    image_names
        .iter()
        .map(|name| {
            Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|stem| by_stem.get(stem).cloned())
        })
        .collect()
}

/// Compare essential bearings against GT centres (`--diagnose-bearing-gt`).
pub(super) fn diagnose_bearing_vs_gt(
    label: &str,
    pairwise: &[PairwiseMatches],
    features: &[FeatureSet],
    camera: &Camera,
    image_names: &[String],
    gt_by_stem: &HashMap<String, Pose>,
    stem_filter: &HashSet<&str>,
) {
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let mut rows: Vec<(String, usize, f64, f64, f64, bool)> = Vec::new();
    for pair in pairwise {
        if !stem_filter.is_empty()
            && !stem_filter.contains(stem_of(pair.image_i))
            && !stem_filter.contains(stem_of(pair.image_j))
        {
            continue;
        }
        let e_count = pair.essential_matches.as_ref().map_or(0, |e| e.len());
        // Some verifier winners keep an essential matrix for diagnostics but
        // do not retain an E-inlier index list when F/H won the accepted
        // configuration.  In that case use the accepted correspondence list
        // for pose decomposition instead of silently dropping the edge.
        if e_count < 8 && pair.essential_matrix.is_none() {
            continue;
        }
        let Some(gti) = gt_by_stem.get(stem_of(pair.image_i)) else {
            continue;
        };
        let Some(gtj) = gt_by_stem.get(stem_of(pair.image_j)) else {
            continue;
        };
        let Some(gt_in_i) = gt_bearing_in_prior_frame(gti, gtj) else {
            continue;
        };
        let corrs = pair_correspondences(pair, features, true);
        if corrs.len() < 8 {
            continue;
        }
        let rel = if let Some(essential) = pair.essential_matrix.as_ref() {
            relative_pose_from_essential(essential, &corrs, camera)
        } else {
            RelativePoseEstimator::default().estimate(&corrs, camera)
        };
        let Some(rel) = rel else {
            continue;
        };
        let r = rel.previous_to_current.rotation;
        let t = rel.previous_to_current.translation;
        let gt_rel = gtj.world_to_camera.compose(&gti.world_to_camera.inverse());
        let rotation_error = (r.inverse() * gt_rel.rotation).angle().to_degrees();
        let Some(est) = (-r.inverse().transform_vector(&t)).try_normalize(1e-12) else {
            continue;
        };
        let err_pri = bearing_alignment_error_deg(&est, &gt_in_i);
        let (err_alt, alt_wins) = if let Some((r_alt, t_alt)) = rel.alternate.as_ref() {
            let t_a = t_alt * rel.translation_scale;
            let d_alt: Vector3<f64> = (-r_alt.inverse().transform_vector(&t_a))
                .try_normalize(1e-12)
                .unwrap_or(est);
            let err = bearing_alignment_error_deg(&d_alt, &gt_in_i);
            (err, err + 1e-3 < err_pri)
        } else {
            (f64::NAN, false)
        };
        rows.push((
            format!("{}-{}", stem_of(pair.image_i), stem_of(pair.image_j)),
            e_count,
            err_pri,
            err_alt,
            rotation_error,
            alt_wins,
        ));
    }
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "=== diagnose-bearing-gt ({label}) {} pair(s) ===",
        rows.len()
    );
    let mut alt_wins = 0usize;
    let mut sum_pri = 0.0f64;
    let mut sum_rotation = 0.0f64;
    for (name, e, pri, alt, rotation, wins) in &rows {
        if *wins {
            alt_wins += 1;
        }
        sum_pri += pri;
        sum_rotation += rotation;
        let alt_s = if alt.is_finite() {
            format!(" alt={alt:.1}°{}", if *wins { " *" } else { "" })
        } else {
            String::new()
        };
        println!("  {name} E={e} R={rotation:.1}° pri={pri:.1}°{alt_s}");
    }
    if !rows.is_empty() {
        println!(
            "  summary: mean_R={:.1}° mean_pri={:.1}° alt_would_help={}/{}",
            sum_rotation / rows.len() as f64,
            sum_pri / rows.len() as f64,
            alt_wins,
            rows.len()
        );
    }
}

/// Dump a GT-independent rotation-cycle diagnostic for the verified view graph.
///
/// A pairwise essential estimate is only a local constraint; a wrong
/// façade/chirality solution can still pass its own RANSAC.  For every
/// available triangle `(i,j,k)`, compare `R_jk * R_ij` with `R_ik` and attach
/// the resulting cycle error to all three edges.  This is deliberately an
/// environment-gated diagnostic rather than a mapper policy: the latter
/// needs an explicit A/B threshold and must not silently change legacy tracks.
pub(super) fn dump_rotation_cycle_diagnostics(
    pairwise: &[PairwiseMatches],
    features: &[FeatureSet],
    camera: &Camera,
    image_names: &[String],
) {
    let mut rotations: HashMap<(usize, usize), UnitQuaternion<f64>> = HashMap::new();
    for pair in pairwise {
        let Some(essential) = pair.essential_matrix.as_ref() else {
            continue;
        };
        let corrs = pair_correspondences(pair, features, true);
        let Some(relative) = relative_pose_from_essential(essential, &corrs, camera) else {
            continue;
        };
        let (i, j, rotation) = if pair.image_i < pair.image_j {
            (
                pair.image_i,
                pair.image_j,
                relative.previous_to_current.rotation,
            )
        } else {
            (
                pair.image_j,
                pair.image_i,
                relative.previous_to_current.rotation.inverse(),
            )
        };
        rotations.insert((i, j), rotation);
    }
    let mut edge_errors: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    let n = features.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let Some(r_ij) = rotations.get(&(i, j)).copied() else {
                continue;
            };
            for k in (j + 1)..n {
                let (Some(r_jk), Some(r_ik)) = (
                    rotations.get(&(j, k)).copied(),
                    rotations.get(&(i, k)).copied(),
                ) else {
                    continue;
                };
                let predicted = r_jk * r_ij;
                let error_deg = (predicted.inverse() * r_ik).angle().to_degrees();
                if !error_deg.is_finite() {
                    continue;
                }
                edge_errors.entry((i, j)).or_default().push(error_deg);
                edge_errors.entry((j, k)).or_default().push(error_deg);
                edge_errors.entry((i, k)).or_default().push(error_deg);
            }
        }
    }
    let stem = |idx: usize| {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
            .to_owned()
    };
    let mut rows: Vec<((usize, usize), Vec<f64>)> = edge_errors.into_iter().collect();
    for (_, errors) in &mut rows {
        errors.sort_by(f64::total_cmp);
    }
    rows.sort_by(|a, b| {
        b.1.get(b.1.len() / 2)
            .unwrap_or(&f64::NEG_INFINITY)
            .total_cmp(a.1.get(a.1.len() / 2).unwrap_or(&f64::NEG_INFINITY))
    });
    eprintln!(
        "sfm-debug: rotation cycles edges={} essential_rotations={} triangles={}",
        rows.len(),
        rotations.len(),
        rows.iter().map(|(_, errors)| errors.len()).sum::<usize>() / 3,
    );
    for ((i, j), errors) in rows.iter().take(30) {
        let median = errors[errors.len() / 2];
        let p90 = errors[((errors.len() * 9).saturating_sub(1) / 10).min(errors.len() - 1)];
        let max = *errors.last().unwrap_or(&f64::NAN);
        eprintln!(
            "sfm-debug: rotation-cycle {}-{} triangles={} median={median:.1}deg p90={p90:.1}deg max={max:.1}deg",
            stem(*i),
            stem(*j),
            errors.len(),
        );
    }
    eprintln!("sfm-debug: rotation-cycle lowest-consistency edges:");
    for ((i, j), errors) in rows.iter().rev().take(20) {
        let median = errors[errors.len() / 2];
        let p90 = errors[((errors.len() * 9).saturating_sub(1) / 10).min(errors.len() - 1)];
        eprintln!(
            "sfm-debug: rotation-cycle {}-{} triangles={} median={median:.1}deg p90={p90:.1}deg",
            stem(*i),
            stem(*j),
            errors.len(),
        );
    }
}

fn export_incremental_component_model(
    out_colmap: &Path,
    result: &IncrementalSfmResult,
    features: &[FeatureSet],
    image_names: &[String],
    fallback_camera: &Camera,
    native_keypoints_for_export: Option<&[Vec<Point2<f64>>]>,
    per_image_calibration: Option<&LoadedPerImageCalibration>,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let registered: Vec<usize> = (0..features.len())
        .filter(|&image| result.poses[image].is_some())
        .collect();
    let remap: HashMap<usize, usize> = registered
        .iter()
        .enumerate()
        .map(|(new_index, &old_index)| (old_index, new_index))
        .collect();
    let poses_out: Vec<Pose> = registered
        .iter()
        .map(|&image| result.poses[image].clone().expect("registered pose"))
        .collect();
    let mut features_out: Vec<FeatureSet> = registered
        .iter()
        .map(|&image| features[image].clone())
        .collect();
    if let Some(native_keypoints) = native_keypoints_for_export {
        replace_feature_keypoints_from_native(&mut features_out, &registered, native_keypoints)
            .map_err(std::io::Error::other)?;
    }
    let names_out: Vec<String> = registered
        .iter()
        .map(|&image| image_names[image].clone())
        .collect();
    let landmarks_out: Vec<ExportLandmark> = result
        .tracks
        .iter()
        .map(|track| {
            let observations = track
                .observations
                .iter()
                .filter_map(|&(image, keypoint, pixel)| {
                    remap
                        .get(&image)
                        .map(|&new_image| (new_image, keypoint, pixel))
                })
                .collect();
            (track.position, observations)
        })
        .collect();
    let refined_camera = result.refined_camera.as_ref().unwrap_or(fallback_camera);
    let summary = if let Some(calibration) = per_image_calibration {
        let cameras_out: Vec<Camera> = registered
            .iter()
            .map(|&image| calibration.native_cameras[image].clone())
            .collect();
        write_colmap_reconstruction_for_3dgs_with_cameras(
            out_colmap,
            &cameras_out,
            &poses_out,
            &features_out,
            &landmarks_out,
            |index| names_out[index].clone(),
        )?
    } else {
        write_colmap_reconstruction_for_3dgs(
            out_colmap,
            refined_camera,
            &poses_out,
            &features_out,
            &landmarks_out,
            |index| names_out[index].clone(),
        )?
    };
    Ok((
        summary.frame_count,
        summary.landmark_count,
        summary.observation_count,
    ))
}

pub(super) fn ranked_view_graph_components(
    num_images: usize,
    pairwise: &[PairwiseMatches],
    min_images: usize,
    max_count: usize,
) -> Vec<Vec<usize>> {
    let edges: Vec<(usize, usize)> = pairwise
        .iter()
        .map(|pair| (pair.image_i, pair.image_j))
        .collect();
    let mut components = connected_components(num_images, &edges);
    components.retain(|component| component.len() >= min_images);
    components.sort_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });
    components.truncate(max_count);
    components
}

pub(super) fn map_verified_view_graph_components(
    out_root: &Path,
    min_images: usize,
    max_count: usize,
    camera: &Camera,
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    native_keypoints_for_export: Option<&[Vec<Point2<f64>>]>,
    per_image_calibration: Option<&LoadedPerImageCalibration>,
) -> Result<(), Box<dyn std::error::Error>> {
    let components = ranked_view_graph_components(features.len(), pairwise, min_images, max_count);
    if components.is_empty() {
        return Err(
            format!("no verified view-graph component has at least {min_images} images").into(),
        );
    }

    let mut total_registered = 0usize;
    let mut total_tracks = 0usize;
    let mut total_observations = 0usize;
    let mut weighted_reprojection_sum = 0.0f64;
    let mut manifest = String::from(
        "rank\tsupplied_images\tverified_pairs\tregistered_images\ttracks\tobservations\tmean_reprojection_px\tmodel_dir\n",
    );
    for (rank, component) in components.iter().enumerate() {
        let mut membership = vec![false; features.len()];
        for &image in component {
            membership[image] = true;
        }
        let component_pairs: Vec<PairwiseMatches> = pairwise
            .iter()
            .filter(|pair| membership[pair.image_i] && membership[pair.image_j])
            .cloned()
            .collect();
        let result = incremental_sfm(camera, features, &component_pairs, config)?;
        let component_dir = out_root.join(format!("component-{rank:03}"));
        let (written_images, written_points, written_observations) =
            export_incremental_component_model(
                &component_dir,
                &result,
                features,
                image_names,
                camera,
                native_keypoints_for_export,
                per_image_calibration,
            )?;
        total_registered += written_images;
        total_tracks += written_points;
        total_observations += written_observations;
        weighted_reprojection_sum += result.mean_reprojection_px * written_observations as f64;
        manifest.push_str(&format!(
            "{rank}\t{}\t{}\t{written_images}\t{written_points}\t{written_observations}\t{:.9}\tcomponent-{rank:03}\n",
            component.len(),
            component_pairs.len(),
            result.mean_reprojection_px,
        ));
        println!(
            "component-model: rank={rank} supplied={} pairs={} registered={} tracks={} observations={} mean_reproj={:.3} px out={}",
            component.len(),
            component_pairs.len(),
            written_images,
            written_points,
            written_observations,
            result.mean_reprojection_px,
            component_dir.display(),
        );
    }
    println!(
        "component-model summary: models={} registered={}/{} tracks={} observations={} weighted_mean_reproj={:.3} px (independent gauges; no connected-model claim)",
        components.len(),
        total_registered,
        features.len(),
        total_tracks,
        total_observations,
        weighted_reprojection_sum / total_observations.max(1) as f64,
    );
    manifest.push_str(&format!(
        "# independent_gauges=true models={} registered_images={} supplied_images={} tracks={} observations={} weighted_mean_reprojection_px={:.9}\n",
        components.len(),
        total_registered,
        features.len(),
        total_tracks,
        total_observations,
        weighted_reprojection_sum / total_observations.max(1) as f64,
    ));
    std::fs::write(out_root.join("components.tsv"), manifest)?;
    Ok(())
}
