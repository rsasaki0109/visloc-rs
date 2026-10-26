//! Localize EuRoC camera images against a map saved by
//! `basalt_euroc_online_slam_demo --save-map`.
//!
//! Each image goes through the mapper's own keyframe frontend (Shi-Tomasi
//! corners, intensity-centroid orientation, 256-bit ORB descriptors), so the
//! query descriptors live in the same space as the map's. Query corners are
//! matched to the map's landmark descriptors by Hamming distance (mutual
//! best match is not required; a Lowe ratio and a distance cap are), then
//! P3P RANSAC with Gauss-Newton refinement estimates the camera pose. When
//! `--prior-radius-m` is set and the previous image localized, only the
//! landmarks within that radius of its camera centre are tried first, with
//! the whole map as the fallback.
//!
//! ```text
//! # 1. Map a sequence and save the map.
//! cargo run --release --example basalt_euroc_online_slam_demo -- \
//!   --euroc-dir /data/MH_01_easy \
//!   --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
//!   --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
//!   --out-dir target/mh01_slam --save-map target/mh01_map
//!
//! # 2. Localize another sequence of the same place against it.
//! cargo run --release --example basalt_euroc_localize_demo -- \
//!   --map target/mh01_map --euroc-dir /data/MH_02_easy --out-dir target/mh02_on_mh01
//! ```
//!
//! Outputs, under `--out-dir`:
//!
//! - `localization.tum`: IMU poses of the localized images in the map's world
//!   frame, `timestamp tx ty tz qx qy qz qw` with the image timestamp, the
//!   convention of the SLAM demo's `trajectory_*.tum`.
//! - `per_frame.csv`: per image, success, matches, inliers and latency.

#[path = "basalt_map/map_files.rs"]
mod map_files;

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use visloc_basalt::euroc::{read_camera_image, read_camera_manifest};
use visloc_basalt::mapper::{
    compute_angles, compute_descriptors, detect_keypoints_mapping, localization_camera_id,
    orb_descriptor_to_f32,
};
use visloc_basalt::pyramid::RawU16Image;
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, LocalizationResult, QueryImage};
use visloc_rs::vision::pnp::{GaussNewtonPoseRefiner, P3PGrunert};
use visloc_rs::{
    AllLandmarksSelector, DescriptorMatch, LocalizationConfig, LocalizationPipeline, Matcher,
    PnPRansac, RadiusLandmarkSelector,
};

/// Brute-force Hamming matcher over binary descriptors stored as `0.0`/`1.0`
/// floats (any value `>= 0.5` is a set bit). For such descriptors squared L2
/// distance equals Hamming distance, but packing them to 256-bit words and
/// counting bits is far faster than the float kernel.
#[derive(Debug, Clone)]
struct HammingMatcher {
    /// Largest accepted Hamming distance.
    max_distance: u32,
    /// Accept only when `best < ratio * second_best`.
    ratio: Option<f32>,
}

fn pack_bits(descriptor: &[f32]) -> Vec<u64> {
    let mut words = vec![0_u64; descriptor.len().div_ceil(64)];
    for (bit, value) in descriptor.iter().enumerate() {
        if *value >= 0.5 {
            words[bit / 64] |= 1 << (bit % 64);
        }
    }
    words
}

impl Matcher for HammingMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        let Some(dimension) = query.first().map(Vec::len) else {
            return Vec::new();
        };
        let train_bits = train
            .iter()
            .map(|descriptor| (descriptor.len() == dimension).then(|| pack_bits(descriptor)))
            .collect::<Vec<_>>();
        query
            .par_iter()
            .enumerate()
            .filter_map(|(query_index, descriptor)| {
                if descriptor.len() != dimension {
                    return None;
                }
                let bits = pack_bits(descriptor);
                let (mut best, mut second) = ((u32::MAX, 0_usize), u32::MAX);
                for (train_index, train) in train_bits.iter().enumerate() {
                    let Some(train) = train else { continue };
                    let distance = bits
                        .iter()
                        .zip(train)
                        .map(|(a, b)| (a ^ b).count_ones())
                        .sum::<u32>();
                    if distance < best.0 {
                        second = best.0;
                        best = (distance, train_index);
                    } else if distance < second {
                        second = distance;
                    }
                }
                let (distance, train_index) = best;
                if distance > self.max_distance {
                    return None;
                }
                let second = (second != u32::MAX).then_some(second as f32);
                if let (Some(ratio), Some(second)) = (self.ratio, second) {
                    if distance as f32 >= ratio * second {
                        return None;
                    }
                }
                Some(DescriptorMatch {
                    query_index,
                    train_index,
                    distance: distance as f32,
                    second_best_distance: second,
                    ratio: second.map(|second| distance as f32 / second.max(1.0)),
                    confidence: None,
                })
            })
            .collect()
    }
}

type Pipeline = LocalizationPipeline<HammingMatcher, AllLandmarksSelector, PnPRansac<P3PGrunert>>;

#[derive(Debug, Clone, PartialEq)]
struct Args {
    map: PathBuf,
    euroc_dir: PathBuf,
    out_dir: PathBuf,
    camera: u16,
    max_frames: Option<usize>,
    stride: usize,
    max_points: Option<usize>,
    max_hamming: u32,
    ratio: Option<f32>,
    min_inliers: usize,
    reprojection_px: f64,
    ransac_iterations: usize,
    prior_radius_m: Option<f64>,
}

impl Args {
    fn usage() -> String {
        "usage: basalt_euroc_localize_demo --map DIR --euroc-dir DIR [--out-dir DIR] \
         [--camera N] [--max-frames N] [--stride N] [--max-points N] [--max-hamming N] \
         [--ratio R|none] [--min-inliers N] [--reprojection-px PX] [--ransac-iterations N] \
         [--prior-radius-m M]"
            .into()
    }

    fn parse<I>(arguments: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut map = None;
        let mut euroc_dir = None;
        let mut args = Self {
            map: PathBuf::new(),
            euroc_dir: PathBuf::new(),
            out_dir: PathBuf::from("target/basalt_euroc_localize_demo"),
            camera: 0,
            max_frames: None,
            stride: 1,
            max_points: None,
            max_hamming: 70,
            ratio: Some(0.8),
            // Below 30 inliers wrong poses appear on EuRoC (25 cm to 3.6 m
            // off at 20); see docs/basalt_map_localization.md.
            min_inliers: 30,
            reprojection_px: 3.0,
            ransac_iterations: 1000,
            prior_radius_m: None,
        };
        let mut arguments = arguments.into_iter();
        while let Some(option) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .ok_or_else(|| format!("{option} requires a value"))
            };
            fn number<T: std::str::FromStr>(option: &str, value: String) -> Result<T, String> {
                value
                    .parse()
                    .map_err(|_| format!("{option}: invalid value {value:?}"))
            }
            match option.as_str() {
                "--help" | "-h" => return Err(Self::usage()),
                "--map" => map = Some(PathBuf::from(value()?)),
                "--euroc-dir" => euroc_dir = Some(PathBuf::from(value()?)),
                "--out-dir" => args.out_dir = PathBuf::from(value()?),
                "--camera" => args.camera = number(&option, value()?)?,
                "--max-frames" => args.max_frames = Some(number(&option, value()?)?),
                "--stride" => args.stride = number(&option, value()?)?,
                "--max-points" => args.max_points = Some(number(&option, value()?)?),
                "--max-hamming" => args.max_hamming = number(&option, value()?)?,
                "--ratio" => {
                    let value = value()?;
                    args.ratio = if value == "none" {
                        None
                    } else {
                        Some(number(&option, value)?)
                    };
                }
                "--min-inliers" => args.min_inliers = number(&option, value()?)?,
                "--reprojection-px" => args.reprojection_px = number(&option, value()?)?,
                "--ransac-iterations" => args.ransac_iterations = number(&option, value()?)?,
                "--prior-radius-m" => args.prior_radius_m = Some(number(&option, value()?)?),
                other => return Err(format!("unknown option {other}\n\n{}", Self::usage())),
            }
        }
        args.map = map.ok_or_else(|| format!("--map is required\n\n{}", Self::usage()))?;
        args.euroc_dir =
            euroc_dir.ok_or_else(|| format!("--euroc-dir is required\n\n{}", Self::usage()))?;
        if args.stride == 0 {
            return Err("--stride must be positive".into());
        }
        Ok(args)
    }
}

fn pipeline(args: &Args) -> Pipeline {
    let config = LocalizationConfig {
        ratio: args.ratio,
        ransac_iterations: args.ransac_iterations,
        reprojection_threshold: args.reprojection_px,
        min_inliers: args.min_inliers,
        ..LocalizationConfig::default()
    };
    let ransac = PnPRansac {
        pose_estimator: P3PGrunert,
        pose_refiner: Some(GaussNewtonPoseRefiner::default()),
        iterations: args.ransac_iterations,
        reprojection_threshold: args.reprojection_px,
        seed: 7,
        early_stop_min_iterations: 0,
        early_stop_inlier_ratio: None,
        confidence: Some(0.999),
    };
    let matcher = HammingMatcher {
        max_distance: args.max_hamming,
        ratio: args.ratio,
    };
    LocalizationPipeline::with_pose_estimator(matcher, AllLandmarksSelector, ransac, config)
}

/// The mapper keyframe frontend, with descriptors unpacked like the map's.
fn query_image(
    image: &RawU16Image,
    camera: &Camera,
    max_points: usize,
) -> Result<QueryImage, Box<dyn std::error::Error>> {
    let corners = detect_keypoints_mapping(image, max_points);
    let angles = compute_angles(image, &corners, true);
    let descriptors = compute_descriptors(image, &corners, &angles)?;
    Ok(QueryImage {
        camera: camera.clone(),
        keypoints: corners,
        descriptors: descriptors.iter().map(orb_descriptor_to_f32).collect(),
    })
}

fn accepted(result: &LocalizationResult) -> bool {
    result.success && result.pose.is_some()
}

fn tum_line(timestamp_ns: i64, world_from_imu: &SE3) -> String {
    let t = world_from_imu.translation;
    let q = world_from_imu.rotation.quaternion();
    format!(
        "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
        timestamp_ns as f64 * 1e-9,
        t.x,
        t.y,
        t.z,
        q.i,
        q.j,
        q.k,
        q.w
    )
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let saved = map_files::load(&args.map)?;
    let camera = saved
        .map
        .cameras
        .get(&localization_camera_id(args.camera))
        .cloned()
        .ok_or_else(|| format!("camera {} is not in the map", args.camera))?;
    let imu_from_camera = saved
        .imu_from_camera
        .get(usize::from(args.camera))
        .cloned()
        .ok_or_else(|| format!("map.json has no extrinsics for camera {}", args.camera))?;
    let camera_from_imu = imu_from_camera.inverse();
    let max_points = args.max_points.unwrap_or(saved.max_points);
    eprintln!(
        "map: images={} landmarks={} descriptors={}",
        saved.map.keyframes.len(),
        saved.map.landmarks.len(),
        saved.descriptors.len()
    );

    let camera_dir = args
        .euroc_dir
        .join("mav0")
        .join(format!("cam{}", args.camera));
    let images = read_camera_manifest(&camera_dir)?
        .into_iter()
        .step_by(args.stride)
        .take(args.max_frames.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    let pipeline = pipeline(args);

    let mut tum = String::from("# timestamp tx ty tz qx qy qz qw\n");
    let mut csv = String::from(
        "timestamp_ns,success,used_prior,keypoints,matches,inliers,inlier_ratio,latency_ms\n",
    );
    let mut previous: Option<Pose> = None;
    let mut latencies = Vec::with_capacity(images.len());
    let mut localized = 0_usize;
    for entry in &images {
        let image = read_camera_image(&entry.path)?;
        let started = Instant::now();
        let query = query_image(&image, &camera, max_points)?;
        let prior_result = previous
            .as_ref()
            .zip(args.prior_radius_m)
            .map(|(pose, radius)| {
                pipeline.localize_with_candidate_selector_and_descriptor_store(
                    &query,
                    &saved.map,
                    &saved.descriptors,
                    RadiusLandmarkSelector::new(pose.camera_center_world(), radius),
                )
            });
        let used_prior = prior_result.as_ref().is_some_and(accepted);
        let result = match prior_result {
            Some(result) if used_prior => result,
            _ => pipeline.localize_with_descriptor_store(&query, &saved.map, &saved.descriptors),
        };
        let latency_ms = started.elapsed().as_secs_f64() * 1e3;
        latencies.push(latency_ms);

        let success = accepted(&result);
        if let (true, Some(pose)) = (success, result.pose.as_ref()) {
            let world_from_imu = pose.camera_to_world().compose(&camera_from_imu);
            tum.push_str(&tum_line(entry.timestamp_ns, &world_from_imu));
            localized += 1;
        }
        previous = success.then(|| result.pose.clone()).flatten();
        writeln!(
            csv,
            "{},{},{},{},{},{},{:.4},{:.2}",
            entry.timestamp_ns,
            success,
            used_prior,
            query.keypoints.len(),
            result.match_count,
            result.inlier_count,
            result.inlier_ratio,
            latency_ms
        )?;
    }

    fs::create_dir_all(&args.out_dir)?;
    fs::write(args.out_dir.join("localization.tum"), tum)?;
    fs::write(args.out_dir.join("per_frame.csv"), csv)?;
    latencies.sort_by(f64::total_cmp);
    let median_latency_ms = latencies.get(latencies.len() / 2).copied().unwrap_or(0.0);
    println!(
        "localized {localized}/{} images (median {median_latency_ms:.1} ms/image) -> {}",
        images.len(),
        args.out_dir.display()
    );
    Ok(())
}

fn main() {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_basalt::mapper::{
        MapperImageFeatures, MapperLandmark, MapperObservation, NfrMapper, NfrMapperLandmarkDb,
        TimeCamId,
    };
    use visloc_basalt::vio::landmarks::StereographicDirection;
    use visloc_basalt::{BasaltCalibration, DoubleSphereCamera};

    fn parse(arguments: &[&str]) -> Result<Args, String> {
        Args::parse(arguments.iter().map(|argument| argument.to_string()))
    }

    #[test]
    fn parser_requires_map_and_sequence() {
        assert!(parse(&["--map", "m"]).is_err());
        assert!(parse(&["--euroc-dir", "d"]).is_err());
        assert!(parse(&["--map", "m", "--euroc-dir", "d", "--stride", "0"]).is_err());
        assert!(parse(&["--map", "m", "--euroc-dir", "d", "--bogus"]).is_err());
        let args = parse(&[
            "--map",
            "m",
            "--euroc-dir",
            "d",
            "--ratio",
            "none",
            "--prior-radius-m",
            "4",
            "--camera",
            "1",
        ])
        .expect("parses");
        assert_eq!(args.ratio, None);
        assert_eq!(args.prior_radius_m, Some(4.0));
        assert_eq!(args.camera, 1);
        assert_eq!(args.min_inliers, 30);
    }

    fn descriptor(seed: usize) -> [u8; 32] {
        // Deterministic, well-separated pseudo-random bits per landmark.
        let mut state = (seed as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
    }

    #[test]
    fn hamming_matcher_applies_distance_cap_and_ratio() {
        let unpack = |bits: &[u8; 32]| orb_descriptor_to_f32(bits);
        let train = (0..4)
            .map(|seed| unpack(&descriptor(seed)))
            .collect::<Vec<_>>();
        let mut near = descriptor(2);
        near[0] ^= 0b0000_0111;
        let query = vec![unpack(&near), unpack(&descriptor(99))];
        let matcher = HammingMatcher {
            max_distance: 70,
            ratio: Some(0.8),
        };
        let matches = matcher.match_descriptors(&query, &train);
        assert_eq!(matches.len(), 1);
        assert_eq!((matches[0].query_index, matches[0].train_index), (0, 2));
        assert_eq!(matches[0].distance, 3.0);
    }

    fn euroc_calibration() -> BasaltCalibration {
        let camera = DoubleSphereCamera::new(
            461.2211371727118,
            459.87157074918844,
            367.22196700849764,
            248.4212673781908,
            5.486900730699193e-06,
            0.6571976377067552,
            752,
            480,
        )
        .expect("camera");
        BasaltCalibration {
            t_imu_cam: vec![SE3::new(
                UnitQuaternion::from_euler_angles(-1.5, 0.02, 1.55),
                Vector3::new(-0.02, -0.06, 0.01),
            )],
            cameras: vec![camera],
            resolutions: vec![(752, 480)],
            calib_accel_bias: vec![0.0; 3],
            calib_gyro_bias: vec![0.0; 3],
            imu_update_rate_hz: 200.0,
            accel_noise_std: Vector3::repeat(1.0),
            gyro_noise_std: Vector3::repeat(1.0),
            accel_bias_std: Vector3::repeat(1.0),
            gyro_bias_std: Vector3::repeat(1.0),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        }
    }

    /// Mapper with two keyframes observing a grid of landmarks, each with
    /// its own descriptor.
    fn synthetic_mapper(points: &[Point3<f64>]) -> NfrMapper {
        let calibration = euroc_calibration();
        let camera = calibration.cameras[0];
        let imu_from_camera = calibration.t_imu_cam[0].clone();
        let poses = [
            SE3::identity(),
            SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1),
                Vector3::new(0.0, 0.4, 0.0),
            ),
        ];
        let mut mapper = NfrMapper::default();
        let mut features = BTreeMap::<TimeCamId, MapperImageFeatures>::new();
        let mut landmarks = BTreeMap::new();
        for (track_id, point) in points.iter().enumerate() {
            let mut observations = Vec::new();
            for (frame_id, pose) in poses.iter().enumerate() {
                let image = TimeCamId::new(frame_id as u64, 0);
                let point_camera = pose
                    .compose(&imu_from_camera)
                    .inverse()
                    .transform_point(point);
                let pixel = camera.project(&point_camera).expect("visible");
                let entry = features
                    .entry(image)
                    .or_insert_with(|| MapperImageFeatures {
                        corners: Vec::new(),
                        corner_angles: Vec::new(),
                        descriptors: Vec::new(),
                        rays: Vec::new(),
                        hashes: Vec::new(),
                        bow_vector: Vec::new(),
                    });
                observations.push(MapperObservation {
                    image,
                    feature_id: entry.corners.len() as u64,
                    pixel,
                });
                entry.corners.push(pixel);
                entry.descriptors.push(descriptor(track_id));
            }
            let host_point = imu_from_camera.inverse().transform_point(point);
            landmarks.insert(
                track_id as u64,
                MapperLandmark {
                    track_id: track_id as u64,
                    host: TimeCamId::new(0, 0),
                    second: TimeCamId::new(1, 0),
                    direction: StereographicDirection::from_bearing(host_point.coords)
                        .expect("direction"),
                    inverse_distance: 1.0 / host_point.coords.norm(),
                    observations,
                },
            );
        }
        for (frame_id, pose) in poses.into_iter().enumerate() {
            mapper.frame_poses.insert(frame_id as u64, pose);
            mapper
                .frame_timestamps
                .insert(frame_id as u64, frame_id as i64);
        }
        mapper.calibration = Some(calibration);
        mapper.feature_corners = features;
        mapper.lmdb = NfrMapperLandmarkDb {
            landmarks,
            observations: BTreeMap::new(),
        };
        mapper.lmdb.rebuild_observation_index();
        mapper
    }

    /// Save a map, load it back and localize a query from a new viewpoint:
    /// exercises the camera model fit, pose conventions and descriptor
    /// packing across the on-disk format.
    #[test]
    fn saved_map_localizes_a_new_viewpoint() {
        let calibration = euroc_calibration();
        let imu_from_camera = calibration.t_imu_cam[0].clone();
        let world_from_camera0 = imu_from_camera.clone();
        let points = (0..120)
            .map(|index| {
                let x = -2.0 + 0.35 * (index % 12) as f64;
                let y = -1.2 + 0.3 * (index / 12) as f64;
                let z = 3.0 + 0.5 * ((index * 7) % 5) as f64;
                world_from_camera0.transform_point(&Point3::new(x, y, z))
            })
            .collect::<Vec<_>>();
        let mapper = synthetic_mapper(&points);
        let export = mapper.localization_map().expect("export");

        let map_dir = std::env::temp_dir().join(format!(
            "visloc_basalt_localize_demo_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&map_dir);
        map_files::save(&map_dir, &export, &calibration, 800).expect("save");
        let saved = map_files::load(&map_dir).expect("load");
        fs::remove_dir_all(&map_dir).expect("cleanup");
        assert_eq!(saved.map.landmarks.len(), points.len());
        assert_eq!(saved.max_points, 800);
        assert!(
            (saved.imu_from_camera[0].translation - imu_from_camera.translation).norm() < 1e-12
        );

        // Query IMU pose between and beside the two keyframes; every fifth
        // query keypoint is an outlier with a random descriptor.
        let world_from_imu = SE3::new(
            UnitQuaternion::from_euler_angles(0.05, -0.03, 0.06),
            Vector3::new(0.15, 0.2, -0.1),
        );
        let camera_from_world = world_from_imu.compose(&imu_from_camera).inverse();
        let camera = saved.map.cameras[&localization_camera_id(0)].clone();
        let double_sphere = calibration.cameras[0];
        let mut keypoints = Vec::new();
        let mut descriptors = Vec::new();
        for (index, point) in points.iter().enumerate() {
            let pixel = double_sphere
                .project(&camera_from_world.transform_point(point))
                .expect("visible");
            if index % 5 == 0 {
                keypoints.push(Point2::new(pixel.x + 37.0, pixel.y - 21.0));
                descriptors.push(orb_descriptor_to_f32(&descriptor(1000 + index)));
            } else {
                keypoints.push(pixel);
                descriptors.push(orb_descriptor_to_f32(&descriptor(index)));
            }
        }
        let query = QueryImage {
            camera,
            keypoints,
            descriptors,
        };
        let args = parse(&["--map", "m", "--euroc-dir", "d"]).expect("parses");
        let result =
            pipeline(&args).localize_with_descriptor_store(&query, &saved.map, &saved.descriptors);
        assert!(accepted(&result), "{:?}", result.failure_reason);
        assert!(result.inlier_count >= 90, "inliers {}", result.inlier_count);
        let estimated = result
            .pose
            .expect("pose")
            .camera_to_world()
            .compose(&imu_from_camera.inverse());
        assert!((estimated.translation - world_from_imu.translation).norm() < 1e-3);
        assert!(estimated.rotation.angle_to(&world_from_imu.rotation) < 1e-3);
    }
}
