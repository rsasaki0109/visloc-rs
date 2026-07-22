//! Acceptance-neutral dual-submap scale probe over a DPVO descriptor dump.
//!
//! This tool deliberately has no trajectory/backend input. It reconstructs two
//! windows from stored SuperPoint observations, matches their independently
//! triangulated landmarks, and asks the typed submap aligner whether a full
//! `Sim3` is observable and E-consistent.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3, UnitQuaternion};
use visloc_rs::slam::{
    estimate_submap_sim3_constraint, LocalSubmap, LocalSubmapBuilder, LocalSubmapConfig,
    LocalSubmapQualityConfig, SubmapPointMatch, SubmapSim3AlignmentConfig,
};
use visloc_rs::vision::features::FeatureSet;
use visloc_rs::vision::matching::{BruteForceMatcher, CrossCheckMatcher, DescriptorMatch, Matcher};
use visloc_rs::vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};
use visloc_rs::{Camera, IncrementalSfmConfig, PairwiseMatches};

#[derive(Debug)]
struct Args {
    dump_dir: PathBuf,
    lightglue_dir: Option<PathBuf>,
    vggt_old_points: Option<PathBuf>,
    vggt_new_points: Option<PathBuf>,
    old_anchor: u64,
    new_anchor: u64,
    radius: u64,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: u32,
    height: u32,
}

impl Default for Args {
    fn default() -> Self {
        // EuRoC cam0 intrinsics and resolution in DPVO's RES=8 patch grid.
        Self {
            dump_dir: PathBuf::new(),
            lightglue_dir: None,
            vggt_old_points: None,
            vggt_new_points: None,
            old_anchor: 38,
            new_anchor: 462,
            radius: 16,
            fx: 458.654 / 8.0,
            fy: 457.296 / 8.0,
            cx: 367.215 / 8.0,
            cy: 248.375 / 8.0,
            width: 752 / 8,
            height: 480 / 8,
        }
    }
}

#[derive(Debug, Clone)]
struct DumpFrame {
    arrival: u64,
    features: FeatureSet,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--dump-dir" => args.dump_dir = PathBuf::from(value),
            "--lightglue-dir" => args.lightglue_dir = Some(PathBuf::from(value)),
            "--vggt-old-points" => args.vggt_old_points = Some(PathBuf::from(value)),
            "--vggt-new-points" => args.vggt_new_points = Some(PathBuf::from(value)),
            "--old-anchor" => args.old_anchor = value.parse()?,
            "--new-anchor" => args.new_anchor = value.parse()?,
            "--radius" => args.radius = value.parse()?,
            "--fx" => args.fx = value.parse()?,
            "--fy" => args.fy = value.parse()?,
            "--cx" => args.cx = value.parse()?,
            "--cy" => args.cy = value.parse()?,
            "--width" => args.width = value.parse()?,
            "--height" => args.height = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if args.dump_dir.as_os_str().is_empty() {
        return Err("--dump-dir is required".into());
    }
    if args.old_anchor >= args.new_anchor {
        return Err("--old-anchor must be before --new-anchor".into());
    }
    if args.vggt_old_points.is_some() != args.vggt_new_points.is_some() {
        return Err("--vggt-old-points and --vggt-new-points must be supplied together".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let camera = Camera::pinhole(
        0,
        args.width,
        args.height,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
    );
    let all_frames = load_dump(&args.dump_dir)?;
    let external_matches = args
        .lightglue_dir
        .as_deref()
        .map(load_external_matches)
        .transpose()?;
    let old_frames = select_window(&all_frames, args.old_anchor, args.radius)?;
    let new_frames = select_window(&all_frames, args.new_anchor, args.radius)?;
    println!(
        "probe old_anchor={} new_anchor={} radius={} old_frames={} new_frames={}",
        args.old_anchor,
        args.new_anchor,
        args.radius,
        old_frames.len(),
        new_frames.len()
    );

    let (old_pairs, old_pair_attempts) =
        verified_temporal_pairs(&old_frames, &camera, external_matches.as_ref());
    let (new_pairs, new_pair_attempts) =
        verified_temporal_pairs(&new_frames, &camera, external_matches.as_ref());
    println!(
        "old_pairwise attempted={} verified={} matches={}",
        old_pair_attempts,
        old_pairs.len(),
        old_pairs
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>()
    );
    println!(
        "new_pairwise attempted={} verified={} matches={}",
        new_pair_attempts,
        new_pairs.len(),
        new_pairs
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>()
    );

    let builder = LocalSubmapBuilder::new(LocalSubmapConfig {
        sfm: IncrementalSfmConfig {
            min_seed_matches: 20,
            final_global_ba: true,
            ..IncrementalSfmConfig::default()
        },
        quality: LocalSubmapQualityConfig {
            min_registered_images: 12,
            min_registration_fraction: 0.70,
            min_landmarks: 30,
            min_observations: 90,
            min_median_track_length: 3.0,
            min_median_max_parallax_deg: 2.0,
            max_mean_reprojection_px: 2.0,
        },
    });
    let old_ids = old_frames
        .iter()
        .map(|frame| frame.arrival)
        .collect::<Vec<_>>();
    let new_ids = new_frames
        .iter()
        .map(|frame| frame.arrival)
        .collect::<Vec<_>>();
    let old_features = old_frames
        .iter()
        .map(|frame| frame.features.clone())
        .collect::<Vec<_>>();
    let new_features = new_frames
        .iter()
        .map(|frame| frame.features.clone())
        .collect::<Vec<_>>();
    let old_submap = match builder.build(&camera, &old_ids, &old_features, &old_pairs) {
        Ok(submap) => submap,
        Err(error) => {
            println!("probe_status=rejected side=old error={error:?}");
            return Ok(());
        }
    };
    let new_submap = match builder.build(&camera, &new_ids, &new_features, &new_pairs) {
        Ok(submap) => submap,
        Err(error) => {
            println!("probe_status=rejected side=new error={error:?}");
            return Ok(());
        }
    };
    print_submap_quality("old", &old_submap);
    print_submap_quality("new", &new_submap);

    let old_anchor = frame_by_arrival(&old_frames, args.old_anchor)?;
    let new_anchor = frame_by_arrival(&new_frames, args.new_anchor)?;
    let Some(anchor_verification) = verified_anchor_rotation(
        &old_anchor.features,
        &new_anchor.features,
        &camera,
        external_matches
            .as_ref()
            .and_then(|matches| matches.get(&(args.old_anchor, args.new_anchor))),
    ) else {
        println!("probe_status=rejected stage=anchor_essential");
        return Ok(());
    };
    println!(
        "anchor_essential_inliers={}",
        anchor_verification.inlier_matches.len()
    );

    let (point_matches, target_from_source_rotation) = if let (Some(old_path), Some(new_path)) = (
        args.vggt_old_points.as_deref(),
        args.vggt_new_points.as_deref(),
    ) {
        let old_points = load_anchor_points(old_path)?;
        let new_points = load_anchor_points(new_path)?;
        let matches = indexed_anchor_point_matches(
            &old_points,
            &new_points,
            &anchor_verification.inlier_matches,
        );
        println!(
            "vggt_anchor_bridge essential_inliers={} old_points={} new_points={} lifted_matches={}",
            anchor_verification.inlier_matches.len(),
            old_points.len(),
            new_points.len(),
            matches.len()
        );
        (matches, anchor_verification.rotation)
    } else {
        let matches = anchor_landmark_matches(
            &old_submap,
            &new_submap,
            args.old_anchor,
            args.new_anchor,
            &anchor_verification.inlier_matches,
        );
        println!(
            "anchor_landmark_bridge essential_inliers={} triangulated_matches={}",
            anchor_verification.inlier_matches.len(),
            matches.len()
        );
        let old_anchor_pose = submap_pose(&old_submap, args.old_anchor)?;
        let new_anchor_pose = submap_pose(&new_submap, args.new_anchor)?;
        let rotation = new_anchor_pose.world_to_camera.rotation.inverse()
            * anchor_verification.rotation
            * old_anchor_pose.world_to_camera.rotation;
        (matches, rotation)
    };
    match estimate_submap_sim3_constraint(
        0,
        1,
        &point_matches,
        &target_from_source_rotation,
        &SubmapSim3AlignmentConfig::default(),
    ) {
        Ok(constraint) => println!(
            "probe_status=sim3_pass scale={:.9} inliers={}/{} inlier_ratio={:.6} mean_residual_ratio={:.9} rotation_disagreement_deg={:.6} loo_log_scale_mad={:.9}",
            constraint.target_from_source.scale,
            constraint.inlier_match_indices.len(),
            constraint.correspondence_count,
            constraint.inlier_ratio,
            constraint.mean_residual_ratio,
            constraint.rotation_disagreement_deg,
            constraint.leave_one_out_log_scale_mad,
        ),
        Err(rejection) => println!("probe_status=sim3_rejected diagnostic={rejection:?}"),
    }
    Ok(())
}

fn print_submap_quality(side: &str, submap: &LocalSubmap) {
    let quality = submap.quality;
    println!(
        "{side}_quality registered={}/{} fraction={:.6} landmarks={} observations={} median_track={:.3} median_parallax_deg={:.6} camera_diameter={:.6} mean_reprojection_px={:.6}",
        quality.registered_images,
        quality.requested_images,
        quality.registration_fraction,
        quality.landmarks,
        quality.observations,
        quality.median_track_length,
        quality.median_max_parallax_deg,
        quality.camera_center_diameter,
        quality.mean_reprojection_px,
    );
}

fn frame_by_arrival(frames: &[DumpFrame], arrival: u64) -> Result<&DumpFrame, Box<dyn Error>> {
    frames
        .iter()
        .find(|frame| frame.arrival == arrival)
        .ok_or_else(|| format!("arrival {arrival} is missing from selected window").into())
}

fn submap_pose(submap: &LocalSubmap, arrival: u64) -> Result<&visloc_rs::Pose, Box<dyn Error>> {
    submap
        .frames
        .iter()
        .find(|frame| frame.source_frame_id == arrival)
        .map(|frame| &frame.pose)
        .ok_or_else(|| format!("anchor arrival {arrival} did not register in its submap").into())
}

fn select_window(
    all_frames: &[DumpFrame],
    anchor: u64,
    radius: u64,
) -> Result<Vec<DumpFrame>, Box<dyn Error>> {
    let low = anchor.saturating_sub(radius);
    let high = anchor.saturating_add(radius);
    let selected = all_frames
        .iter()
        .filter(|frame| frame.arrival >= low && frame.arrival <= high)
        .cloned()
        .collect::<Vec<_>>();
    if !selected.iter().any(|frame| frame.arrival == anchor) {
        return Err(format!("anchor {anchor} is missing from descriptor dump").into());
    }
    Ok(selected)
}

const TEMPORAL_OFFSETS: [usize; 6] = [1, 2, 4, 8, 12, 16];
const MATCH_RATIO: f32 = 0.8;

fn verified_temporal_pairs(
    frames: &[DumpFrame],
    camera: &Camera,
    external_matches: Option<&HashMap<(u64, u64), Vec<(usize, usize)>>>,
) -> (Vec<PairwiseMatches>, usize) {
    let matcher = CrossCheckMatcher::new(BruteForceMatcher {
        ratio: Some(MATCH_RATIO),
    });
    let mut pairwise = Vec::new();
    let mut attempts = 0;
    for image_i in 0..frames.len() {
        for offset in TEMPORAL_OFFSETS {
            let image_j = image_i + offset;
            if image_j >= frames.len() {
                continue;
            }
            attempts += 1;
            let provided = external_matches.and_then(|matches| {
                matches.get(&(frames[image_i].arrival, frames[image_j].arrival))
            });
            let verified = if let Some(matches) = provided {
                verified_index_matches(
                    &frames[image_i].features,
                    &frames[image_j].features,
                    camera,
                    matches,
                    20,
                )
            } else if external_matches.is_none() {
                verified_pair_matches(
                    &frames[image_i].features,
                    &frames[image_j].features,
                    camera,
                    &matcher,
                    20,
                )
            } else {
                None
            };
            if let Some(matches) = verified {
                pairwise.push(PairwiseMatches {
                    image_i,
                    image_j,
                    matches,
                });
            }
        }
    }
    (pairwise, attempts)
}

fn verified_pair_matches(
    first: &FeatureSet,
    second: &FeatureSet,
    camera: &Camera,
    matcher: &CrossCheckMatcher<BruteForceMatcher>,
    min_inliers: usize,
) -> Option<Vec<(usize, usize)>> {
    let matches: Vec<DescriptorMatch> =
        matcher.match_descriptors(&first.descriptors, &second.descriptors);
    let kept = matches
        .into_iter()
        .map(|descriptor_match| (descriptor_match.query_index, descriptor_match.train_index))
        .collect::<Vec<_>>();
    verified_index_matches(first, second, camera, &kept, min_inliers)
}

fn verified_index_matches(
    first: &FeatureSet,
    second: &FeatureSet,
    camera: &Camera,
    matches: &[(usize, usize)],
    min_inliers: usize,
) -> Option<Vec<(usize, usize)>> {
    let mut kept = Vec::new();
    let mut correspondences = Vec::new();
    for &(query_index, train_index) in matches {
        let (Some(&previous_xy), Some(&current_xy)) = (
            first.keypoints.get(query_index),
            second.keypoints.get(train_index),
        ) else {
            continue;
        };
        kept.push((query_index, train_index));
        correspondences.push(TwoViewCorrespondence::new(previous_xy, current_xy));
    }
    let relative = RelativePoseEstimator::default().estimate(&correspondences, camera)?;
    if relative.inliers.len() < min_inliers {
        return None;
    }
    Some(relative.inliers.iter().map(|index| kept[*index]).collect())
}

struct AnchorVerification {
    rotation: UnitQuaternion<f64>,
    inlier_matches: Vec<(usize, usize)>,
}

fn verified_anchor_rotation(
    old: &FeatureSet,
    new: &FeatureSet,
    camera: &Camera,
    external_matches: Option<&Vec<(usize, usize)>>,
) -> Option<AnchorVerification> {
    let owned_matches;
    let matches = if let Some(matches) = external_matches {
        matches.as_slice()
    } else {
        let matcher = CrossCheckMatcher::new(BruteForceMatcher {
            ratio: Some(MATCH_RATIO),
        });
        owned_matches = matcher
            .match_descriptors(&old.descriptors, &new.descriptors)
            .into_iter()
            .map(|descriptor_match| (descriptor_match.query_index, descriptor_match.train_index))
            .collect::<Vec<_>>();
        &owned_matches
    };
    let inlier_matches = verified_index_matches(old, new, camera, matches, 30)?;
    let correspondences = inlier_matches
        .iter()
        .map(|&(old_index, new_index)| {
            TwoViewCorrespondence::new(old.keypoints[old_index], new.keypoints[new_index])
        })
        .collect::<Vec<_>>();
    let relative = RelativePoseEstimator::default().estimate(&correspondences, camera)?;
    Some(AnchorVerification {
        rotation: relative.previous_to_current.rotation,
        inlier_matches,
    })
}

fn anchor_landmark_matches(
    old_submap: &LocalSubmap,
    new_submap: &LocalSubmap,
    old_anchor: u64,
    new_anchor: u64,
    inlier_matches: &[(usize, usize)],
) -> Vec<SubmapPointMatch> {
    let old_observations = landmark_observation_lookup(old_submap);
    let new_observations = landmark_observation_lookup(new_submap);
    let mut used_old = HashSet::new();
    let mut used_new = HashSet::new();
    let mut point_matches = Vec::new();
    for &(old_keypoint, new_keypoint) in inlier_matches {
        let (Some(&old_index), Some(&new_index)) = (
            old_observations.get(&(old_anchor, old_keypoint)),
            new_observations.get(&(new_anchor, new_keypoint)),
        ) else {
            continue;
        };
        if !used_old.insert(old_index) || !used_new.insert(new_index) {
            continue;
        }
        let old_landmark = &old_submap.landmarks[old_index];
        let new_landmark = &new_submap.landmarks[new_index];
        point_matches.push(SubmapPointMatch {
            source_landmark_id: old_landmark.local_landmark_id,
            target_landmark_id: new_landmark.local_landmark_id,
            source_point: old_landmark.position,
            target_point: new_landmark.position,
        });
    }
    point_matches
}

fn landmark_observation_lookup(submap: &LocalSubmap) -> HashMap<(u64, usize), usize> {
    let mut lookup = HashMap::new();
    for (landmark_index, landmark) in submap.landmarks.iter().enumerate() {
        for observation in &landmark.observations {
            lookup.insert(
                (observation.source_frame_id, observation.keypoint_index),
                landmark_index,
            );
        }
    }
    lookup
}

fn load_anchor_points(path: &Path) -> Result<HashMap<usize, Point3<f64>>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut points = HashMap::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let columns = line.split_whitespace().collect::<Vec<_>>();
        if columns.len() != 5 {
            return Err(format!(
                "{}:{} expected five columns",
                path.display(),
                line_index + 1
            )
            .into());
        }
        let keypoint_index = columns[0].parse::<usize>()?;
        let point = Point3::new(
            columns[1].parse::<f64>()?,
            columns[2].parse::<f64>()?,
            columns[3].parse::<f64>()?,
        );
        if !point.coords.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "{}:{} contains a non-finite point",
                path.display(),
                line_index + 1
            )
            .into());
        }
        if points.insert(keypoint_index, point).is_some() {
            return Err(
                format!("{} has duplicate keypoint {keypoint_index}", path.display()).into(),
            );
        }
    }
    Ok(points)
}

fn indexed_anchor_point_matches(
    old_points: &HashMap<usize, Point3<f64>>,
    new_points: &HashMap<usize, Point3<f64>>,
    inlier_matches: &[(usize, usize)],
) -> Vec<SubmapPointMatch> {
    inlier_matches
        .iter()
        .filter_map(|&(old_index, new_index)| {
            Some(SubmapPointMatch {
                source_landmark_id: old_index as u64,
                target_landmark_id: new_index as u64,
                source_point: *old_points.get(&old_index)?,
                target_point: *new_points.get(&new_index)?,
            })
        })
        .collect()
}

fn load_external_matches(
    dir: &Path,
) -> Result<HashMap<(u64, u64), Vec<(usize, usize)>>, Box<dyn Error>> {
    let manifest = fs::read_to_string(dir.join("manifest.csv"))?;
    let mut pairs = HashMap::new();
    for (line_index, line) in manifest.lines().enumerate().skip(1) {
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() != 5 {
            return Err(format!(
                "external manifest line {} has {} columns",
                line_index + 1,
                columns.len()
            )
            .into());
        }
        let arrival_i = columns[1].parse::<u64>()?;
        let arrival_j = columns[2].parse::<u64>()?;
        let text = fs::read_to_string(dir.join(columns[3]))?;
        let mut matches = Vec::new();
        for (match_line_index, match_line) in text.lines().enumerate() {
            let match_line = match_line.trim();
            if match_line.is_empty() || match_line.starts_with('#') {
                continue;
            }
            let fields = match_line.split_whitespace().collect::<Vec<_>>();
            if fields.len() < 2 {
                return Err(format!(
                    "{}:{} has fewer than two columns",
                    columns[3],
                    match_line_index + 1
                )
                .into());
            }
            matches.push((fields[0].parse()?, fields[1].parse()?));
        }
        if pairs.insert((arrival_i, arrival_j), matches).is_some() {
            return Err(format!("duplicate external pair {arrival_i}/{arrival_j}").into());
        }
    }
    println!(
        "external_matcher=lightglue directory={} pairs={}",
        dir.display(),
        pairs.len()
    );
    Ok(pairs)
}

fn load_dump(dir: &Path) -> Result<Vec<DumpFrame>, Box<dyn Error>> {
    let manifest = fs::read_to_string(dir.join("manifest.csv"))?;
    let mut frames = Vec::new();
    for (line_index, line) in manifest.lines().enumerate().skip(1) {
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() != 5 {
            return Err(format!(
                "manifest line {} has {} columns",
                line_index + 1,
                columns.len()
            )
            .into());
        }
        let arrival: u64 = columns[0].parse()?;
        let keypoint_count: usize = columns[1].parse()?;
        let descriptor_dim: usize = columns[2].parse()?;
        let keypoint_data = read_npy_f32(&dir.join(columns[3]), keypoint_count * 2)?;
        let descriptor_data = read_npy_f32(&dir.join(columns[4]), keypoint_count * descriptor_dim)?;
        let keypoints = keypoint_data
            .chunks_exact(2)
            .map(|xy| Point2::new(xy[0] as f64, xy[1] as f64))
            .collect::<Vec<_>>();
        let descriptors = descriptor_data
            .chunks_exact(descriptor_dim)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        frames.push(DumpFrame {
            arrival,
            features: FeatureSet::new(keypoints, descriptors)?,
        });
    }
    frames.sort_by_key(|frame| frame.arrival);
    Ok(frames)
}

fn read_npy_f32(path: &Path, expected_values: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a numpy array", path.display()).into());
    }
    if bytes[6] != 1 || bytes[7] != 0 {
        return Err(format!("{} is not numpy v1.0", path.display()).into());
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_offset = 10usize
        .checked_add(header_len)
        .ok_or("npy header length overflow")?;
    let expected_bytes = expected_values
        .checked_mul(4)
        .ok_or("npy value count overflow")?;
    if bytes.len() != data_offset + expected_bytes {
        return Err(format!(
            "{} payload size {} != expected {}",
            path.display(),
            bytes.len().saturating_sub(data_offset),
            expected_bytes
        )
        .into());
    }
    Ok(bytes[data_offset..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
