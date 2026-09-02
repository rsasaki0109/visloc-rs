//! Replay a verified-pair snapshot through the generalized-rig mapper.
//!
//! The manifest is deliberately dependency-free and explicit:
//!
//! ```text
//! # S index camera_id width height fx fy cx cy qw qx qy qz tx ty tz
//! S 0 1 848 800 284.98 286.10 425.24 398.46 1 0 0 0 0 0 0
//! S 1 2 848 800 284.81 285.97 427.66 397.12 0.99999 ... -0.0639 ...
//! # F frame_id image_name sensor_index
//! F 0 cam1_000000.png 0
//! F 0 cam2_000001.png 1
//! ```
//!
//! Sensor poses are `T_sensor<-rig`. Feature files are resolved from each
//! image stem plus `--feature-suffix`; descriptors are intentionally not kept
//! because track construction and mapping consume only keypoints and verified
//! feature indices.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nalgebra::{Point2, Quaternion, UnitQuaternion, Vector3};
use visloc_rs::verified_pair_snapshot;
use visloc_rs::{
    incremental_rig_sfm, write_colmap_reconstruction_for_3dgs_with_cameras, Camera, FeatureSet,
    GeneralizedCameraRig, PairwiseMatches, RigFrame, RigFrameImage, RigSensor, RigSfmConfig,
    RobustKernel, SE3,
};

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    features_dir: PathBuf,
    feature_suffix: String,
    snapshot: PathBuf,
    out_colmap: PathBuf,
    min_pnp_inliers: usize,
    max_reprojection_error_px: f64,
    pnp_max_iterations: usize,
    final_bundle_adjustment: bool,
    max_matches_per_pair: Option<usize>,
    local_ba_every: usize,
    local_ba_window_size: usize,
    local_ba_iterations: usize,
    final_ba_passes: usize,
    final_ba_window_size: usize,
    final_ba_fix_window_ends: bool,
    ransac_seed: u64,
    ba_huber_delta: f64,
    structure_refinement_iterations: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut values = std::env::args().skip(1);
    let mut manifest = None;
    let mut features_dir = None;
    let mut feature_suffix = "_features.txt".to_owned();
    let mut snapshot = None;
    let mut out_colmap = None;
    let mut min_pnp_inliers = 8usize;
    let mut max_reprojection_error_px = 4.0;
    let mut pnp_max_iterations = 512usize;
    let defaults = RigSfmConfig::default();
    let mut final_bundle_adjustment = defaults.final_bundle_adjustment;
    let mut max_matches_per_pair = None;
    let mut local_ba_every = defaults.local_ba_every;
    let mut local_ba_window_size = defaults.local_ba_window_size;
    let mut local_ba_iterations = defaults.local_ba_iterations;
    let mut final_ba_passes = defaults.final_ba_passes;
    let mut final_ba_window_size = defaults.final_ba_window_size;
    let mut final_ba_fix_window_ends = defaults.final_ba_fix_window_ends;
    let mut ransac_seed = defaults.ransac_seed;
    let mut ba_huber_delta = 6.0;
    let mut structure_refinement_iterations = defaults.structure_refinement_iterations;
    while let Some(flag) = values.next() {
        let mut value = || {
            values
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value()?)),
            "--features-dir" => features_dir = Some(PathBuf::from(value()?)),
            "--feature-suffix" => feature_suffix = value()?,
            "--snapshot" => snapshot = Some(PathBuf::from(value()?)),
            "--out-colmap" => out_colmap = Some(PathBuf::from(value()?)),
            "--min-pnp-inliers" => {
                min_pnp_inliers = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--max-reprojection-error-px" => {
                max_reprojection_error_px = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--pnp-max-iterations" => {
                pnp_max_iterations = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--no-final-ba" => final_bundle_adjustment = false,
            "--final-ba" => final_bundle_adjustment = true,
            "--max-matches-per-pair" => {
                let parsed: usize = value()?.parse().map_err(|error| format!("{error}"))?;
                max_matches_per_pair = (parsed > 0).then_some(parsed);
            }
            "--local-ba-every" => {
                local_ba_every = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--local-ba-window" => {
                local_ba_window_size = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--local-ba-iterations" => {
                local_ba_iterations = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--final-ba-passes" => {
                final_ba_passes = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--final-ba-window" => {
                final_ba_window_size = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--final-ba-single-anchor" => final_ba_fix_window_ends = false,
            "--final-ba-fix-window-ends" => final_ba_fix_window_ends = true,
            "--ransac-seed" => {
                ransac_seed = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--ba-huber-delta" => {
                ba_huber_delta = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--structure-refinement-iterations" => {
                structure_refinement_iterations =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--help" | "-h" => {
                return Err(concat!(
                    "usage: generalized_rig_sfm --manifest FILE --features-dir DIR ",
                    "--snapshot FILE --out-colmap DIR [--feature-suffix _features.txt] ",
                    "[--min-pnp-inliers 8] [--max-reprojection-error-px 4] ",
                    "[--pnp-max-iterations 512] [--max-matches-per-pair 0] ",
                    "[--local-ba-every 10] [--local-ba-window 40] ",
                    "[--local-ba-iterations 8] [--ba-huber-delta 6] ",
                    "[--structure-refinement-iterations 5] ",
                    "[--ransac-seed 7] [--final-ba|--no-final-ba] ",
                    "[--final-ba-passes 2] [--final-ba-window 60]",
                    " [--final-ba-fix-window-ends|--final-ba-single-anchor]"
                )
                .into());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if feature_suffix.is_empty() {
        return Err("--feature-suffix must not be empty".into());
    }
    Ok(Args {
        manifest: manifest.ok_or("--manifest is required")?,
        features_dir: features_dir.ok_or("--features-dir is required")?,
        feature_suffix,
        snapshot: snapshot.ok_or("--snapshot is required")?,
        out_colmap: out_colmap.ok_or("--out-colmap is required")?,
        min_pnp_inliers,
        max_reprojection_error_px,
        pnp_max_iterations,
        final_bundle_adjustment,
        max_matches_per_pair,
        local_ba_every,
        local_ba_window_size,
        local_ba_iterations,
        final_ba_passes,
        final_ba_window_size,
        final_ba_fix_window_ends,
        ransac_seed,
        ba_huber_delta,
        structure_refinement_iterations,
    })
}

#[derive(Debug)]
struct ParsedManifest {
    rig: GeneralizedCameraRig,
    frame_rows: Vec<(u64, String, usize)>,
}

fn token<'a>(tokens: &'a [&str], index: usize, line: usize) -> Result<&'a str, String> {
    tokens
        .get(index)
        .copied()
        .ok_or_else(|| format!("manifest line {line}: missing field {index}"))
}

fn number<T: std::str::FromStr>(tokens: &[&str], index: usize, line: usize) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    token(tokens, index, line)?
        .parse()
        .map_err(|error| format!("manifest line {line} field {index}: {error}"))
}

fn parse_manifest(path: &Path) -> Result<ParsedManifest, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    let mut sensors = BTreeMap::new();
    let mut frame_rows = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let line = zero_line + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        match token(&tokens, 0, line)? {
            "S" => {
                if tokens.len() != 16 {
                    return Err(format!(
                        "manifest line {line}: sensor row requires 16 fields, found {}",
                        tokens.len()
                    ));
                }
                let index: usize = number(&tokens, 1, line)?;
                let camera = Camera::pinhole(
                    number(&tokens, 2, line)?,
                    number(&tokens, 3, line)?,
                    number(&tokens, 4, line)?,
                    number(&tokens, 5, line)?,
                    number(&tokens, 6, line)?,
                    number(&tokens, 7, line)?,
                    number(&tokens, 8, line)?,
                );
                let quaternion: Quaternion<f64> = Quaternion::new(
                    number(&tokens, 9, line)?,
                    number(&tokens, 10, line)?,
                    number(&tokens, 11, line)?,
                    number(&tokens, 12, line)?,
                );
                let norm = quaternion.norm();
                if !norm.is_finite() || norm <= 1.0e-12 {
                    return Err(format!("manifest line {line}: invalid sensor quaternion"));
                }
                let sensor = RigSensor {
                    camera,
                    sensor_from_rig: SE3::new(
                        UnitQuaternion::new_normalize(quaternion),
                        Vector3::new(
                            number(&tokens, 13, line)?,
                            number(&tokens, 14, line)?,
                            number(&tokens, 15, line)?,
                        ),
                    ),
                };
                if sensors.insert(index, sensor).is_some() {
                    return Err(format!("manifest line {line}: duplicate sensor {index}"));
                }
            }
            "F" => {
                if tokens.len() != 4 {
                    return Err(format!(
                        "manifest line {line}: frame row requires 4 fields, found {}",
                        tokens.len()
                    ));
                }
                frame_rows.push((
                    number(&tokens, 1, line)?,
                    token(&tokens, 2, line)?.to_owned(),
                    number(&tokens, 3, line)?,
                ));
            }
            kind => return Err(format!("manifest line {line}: unknown row kind {kind}")),
        }
    }
    if sensors.len() < 2 || frame_rows.is_empty() {
        return Err("manifest requires at least two sensors and one frame row".into());
    }
    if sensors.keys().copied().ne(0..sensors.len()) {
        return Err("manifest sensor indices must be contiguous from zero".into());
    }
    let rig = GeneralizedCameraRig::new(sensors.into_values().collect())
        .ok_or("manifest contains invalid rig calibration")?;
    Ok(ParsedManifest { rig, frame_rows })
}

fn feature_path(directory: &Path, image_name: &str, suffix: &str) -> Result<PathBuf, String> {
    let stem = Path::new(image_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("image name has no UTF-8 stem: {image_name:?}"))?;
    Ok(directory.join(format!("{stem}{suffix}")))
}

fn read_keypoints(path: &Path) -> Result<FeatureSet, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read feature file {}: {error}", path.display()))?;
    let mut keypoints = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let mut fields = text.split_whitespace();
        let x = fields
            .next()
            .ok_or_else(|| format!("{}:{} missing x", path.display(), zero_line + 1))?
            .parse::<f64>()
            .map_err(|error| format!("{}:{} invalid x: {error}", path.display(), zero_line + 1))?;
        let y = fields
            .next()
            .ok_or_else(|| format!("{}:{} missing y", path.display(), zero_line + 1))?
            .parse::<f64>()
            .map_err(|error| format!("{}:{} invalid y: {error}", path.display(), zero_line + 1))?;
        keypoints.push(Point2::new(x, y));
    }
    let descriptors = vec![Vec::new(); keypoints.len()];
    FeatureSet::new(keypoints, descriptors).map_err(|error| error.to_string())
}

fn build_frames(
    rows: &[(u64, String, usize)],
    image_names: &[String],
) -> Result<Vec<RigFrame>, String> {
    let image_indices = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut grouped: BTreeMap<u64, Vec<RigFrameImage>> = BTreeMap::new();
    for (frame_id, image_name, sensor_index) in rows {
        let image_index = image_indices
            .get(image_name.as_str())
            .copied()
            .ok_or_else(|| format!("manifest image {image_name:?} is absent from snapshot"))?;
        grouped.entry(*frame_id).or_default().push(RigFrameImage {
            image_index,
            sensor_index: *sensor_index,
        });
    }
    let assigned = grouped.values().map(Vec::len).sum::<usize>();
    if assigned != image_names.len() {
        return Err(format!(
            "manifest assigns {assigned} images but snapshot contains {}",
            image_names.len()
        ));
    }
    Ok(grouped
        .into_values()
        .map(|images| RigFrame { images })
        .collect())
}

fn convert_pairs(
    snapshot: &verified_pair_snapshot::Snapshot,
) -> Result<Vec<PairwiseMatches>, String> {
    snapshot
        .pairs
        .iter()
        .map(|pair| {
            let image_i = usize::try_from(pair.image_i)
                .map_err(|_| "snapshot image_i does not fit usize".to_owned())?;
            let image_j = usize::try_from(pair.image_j)
                .map_err(|_| "snapshot image_j does not fit usize".to_owned())?;
            let matches = pair
                .matches
                .iter()
                .map(|&(left, right)| {
                    Ok((
                        usize::try_from(left)
                            .map_err(|_| "snapshot left keypoint does not fit usize".to_owned())?,
                        usize::try_from(right)
                            .map_err(|_| "snapshot right keypoint does not fit usize".to_owned())?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(PairwiseMatches::new(image_i, image_j, matches))
        })
        .collect()
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix("VmHWM:")?;
        rest.split_whitespace().next()?.parse().ok()
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args().map_err(std::io::Error::other)?;
    let started = Instant::now();
    let manifest = parse_manifest(&args.manifest).map_err(std::io::Error::other)?;
    let snapshot = verified_pair_snapshot::read_mapper_compact(&args.snapshot)
        .map_err(std::io::Error::other)?;
    let frames =
        build_frames(&manifest.frame_rows, &snapshot.image_names).map_err(std::io::Error::other)?;
    let features = snapshot
        .image_names
        .iter()
        .map(|name| {
            feature_path(&args.features_dir, name, &args.feature_suffix)
                .and_then(|path| read_keypoints(&path))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(std::io::Error::other)?;
    for (index, (&expected, feature)) in snapshot.feature_counts.iter().zip(&features).enumerate() {
        if expected != feature.len() as u64 {
            return Err(format!(
                "feature count mismatch at image {index}: snapshot={expected}, file={}",
                feature.len()
            )
            .into());
        }
    }
    let mut pairs = convert_pairs(&snapshot).map_err(std::io::Error::other)?;
    if let Some(limit) = args.max_matches_per_pair {
        for pair in &mut pairs {
            pair.matches.truncate(limit);
        }
    }
    eprintln!(
        "rig-replay input: frames={} images={} pairs={} matches={} keypoints={} VmHWM={} KiB",
        frames.len(),
        features.len(),
        pairs.len(),
        pairs.iter().map(|pair| pair.matches.len()).sum::<usize>(),
        features.iter().map(FeatureSet::len).sum::<usize>(),
        peak_rss_kib().unwrap_or(0),
    );
    let mapping_started = Instant::now();
    let result = incremental_rig_sfm(
        &manifest.rig,
        &frames,
        &features,
        &pairs,
        &RigSfmConfig {
            min_pnp_inliers: args.min_pnp_inliers,
            max_reprojection_error_px: args.max_reprojection_error_px,
            pnp_max_iterations: args.pnp_max_iterations,
            final_bundle_adjustment: args.final_bundle_adjustment,
            local_ba_every: args.local_ba_every,
            local_ba_window_size: args.local_ba_window_size,
            local_ba_iterations: args.local_ba_iterations,
            final_ba_passes: args.final_ba_passes,
            final_ba_window_size: args.final_ba_window_size,
            final_ba_fix_window_ends: args.final_ba_fix_window_ends,
            ransac_seed: args.ransac_seed,
            structure_refinement_iterations: args.structure_refinement_iterations,
            ba_config: visloc_rs::BaConfig {
                robust_kernel: RobustKernel::Huber {
                    delta: args.ba_huber_delta,
                },
                ..RigSfmConfig::default().ba_config
            },
            ..RigSfmConfig::default()
        },
    )?;
    let mapper_seconds = mapping_started.elapsed().as_secs_f64();
    if let Some(ba) = result.bundle_adjustment {
        eprintln!(
            "rig-replay BA: observations={} cost={:.9}->{:.9} iterations={} converged={}",
            ba.observations, ba.initial_cost, ba.final_cost, ba.iterations, ba.converged,
        );
    }

    let registered = result
        .image_poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.as_ref().map(|pose| (image, pose.clone())))
        .collect::<Vec<_>>();
    let remap = registered
        .iter()
        .enumerate()
        .map(|(output, (input, _))| (*input, output))
        .collect::<HashMap<_, _>>();
    let cameras = registered
        .iter()
        .map(|(image, _)| {
            let sensor_index = frames
                .iter()
                .flat_map(|frame| &frame.images)
                .find(|row| row.image_index == *image)
                .expect("validated image assignment")
                .sensor_index;
            manifest.rig.sensors()[sensor_index].camera.clone()
        })
        .collect::<Vec<_>>();
    let poses = registered
        .iter()
        .map(|(_, pose)| pose.clone())
        .collect::<Vec<_>>();
    let names = registered
        .iter()
        .map(|(image, _)| snapshot.image_names[*image].clone())
        .collect::<Vec<_>>();
    let output_features = registered
        .iter()
        .map(|(image, _)| features[*image].clone())
        .collect::<Vec<_>>();
    let landmarks = result
        .tracks
        .iter()
        .map(|track| {
            let observations = track
                .observations
                .iter()
                .filter_map(|&(image, keypoint, pixel)| {
                    remap.get(&image).map(|output| (*output, keypoint, pixel))
                })
                .collect::<Vec<_>>();
            (track.position, observations)
        })
        .collect::<Vec<_>>();
    let export = write_colmap_reconstruction_for_3dgs_with_cameras(
        &args.out_colmap,
        &cameras,
        &poses,
        &output_features,
        &landmarks,
        |index| names[index].clone(),
    )?;
    println!(
        concat!(
            "rig-replay result: registered_frames={}/{} registered_images={}/{} ",
            "tracks={} observations={} mean_reprojection_px={:.9} seed_frame={} ",
            "mapper_seconds={:.6} total_seconds={:.6} VmHWM_KiB={} ",
            "triangulation_attempts={} cache_insertions={} pnp_attempts={} local_ba_runs={} structure_refined_tracks={} out={}"
        ),
        result.registered_frames,
        frames.len(),
        result.registered_images,
        features.len(),
        export.landmark_count,
        export.observation_count,
        result.mean_reprojection_error_px,
        result.seed_frame_index,
        mapper_seconds,
        started.elapsed().as_secs_f64(),
        peak_rss_kib().unwrap_or(0),
        result.work.triangulation_attempts,
        result.work.correspondence_cache_insertions,
        result.work.pnp_attempts,
        result.work.local_ba_runs,
        result.work.structure_refined_tracks,
        args.out_colmap.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_manifest_and_preserves_frame_grouping() {
        let root = std::env::temp_dir().join(format!(
            "visloc-generalized-rig-manifest-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("rig.txt");
        std::fs::write(
            &path,
            concat!(
                "S 0 1 848 800 285 286 425 398 1 0 0 0 0 0 0\n",
                "S 1 2 848 800 284 286 428 397 1 0 0 0 -0.2 0 0\n",
                "F 10 left.png 0\n",
                "F 10 right.png 1\n",
            ),
        )
        .unwrap();
        let parsed = parse_manifest(&path).unwrap();
        assert_eq!(parsed.rig.sensors().len(), 2);
        let frames = build_frames(
            &parsed.frame_rows,
            &["left.png".to_owned(), "right.png".to_owned()],
        )
        .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].images.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
