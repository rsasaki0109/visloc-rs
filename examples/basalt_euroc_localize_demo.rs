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
//! With `--vio --calibration FILE --config FILE`, the stereo-inertial Basalt
//! VIO also runs on the sequence, and every `--localize-every`-th image is
//! localized. Accepted fixes anchor the VIO to the map
//! (`visloc_localization::MapAnchor`: a gravity-aligned yaw + translation
//! offset, gated against outliers), so every frame gets a map-frame pose once
//! the anchor is set, including the frames the map does not localize.
//!
//! ```text
//! cargo run --release --example basalt_euroc_localize_demo -- \
//!   --map target/mh01_map --euroc-dir /data/MH_03_medium --out-dir target/mh03_vio \
//!   --vio --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
//!   --config configs/basalt/variants/official_euroc_ds/euroc_config.json
//! ```
//!
//! Outputs, under `--out-dir`:
//!
//! - `localization.tum`: IMU poses of the localized images in the map's world
//!   frame, `timestamp tx ty tz qx qy qz qw` with the image timestamp, the
//!   convention of the SLAM demo's `trajectory_*.tum`.
//! - `per_frame.csv`: per localization attempt, success, matches, inliers,
//!   latency and, with `--vio`, what the anchor did with the fix.
//! - With `--vio`: `trajectory_map.tum`, every anchored frame's IMU pose in
//!   the map frame; `trajectory_map.csv`, the same poses with the time and
//!   VIO distance since the anchor's newest fix (the fused error grows with
//!   them, so use them to judge a pose); `trajectory_vio.tum`, the raw VIO in
//!   its own frame.
//! - With `--dump-inliers FILE`: per accepted localization, one CSV row per
//!   PnP inlier, `timestamp_ns,x,y,landmark_id` (the query keypoint in
//!   pixels and the map landmark it matched), e.g. for drawing the matches.

#[path = "basalt_map/map_files.rs"]
mod map_files;

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::Point2;
use visloc_basalt::euroc::{read_camera_image, read_camera_manifest};
use visloc_basalt::mapper::{
    compute_angles, compute_descriptors, detect_keypoints_mapping, localization_camera_id,
    orb_descriptor_to_f32,
};
use visloc_basalt::pyramid::RawU16Image;
use visloc_basalt::{BasaltVioEstimatorAdapter, EurocSensorDataset};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, LocalizationResult, QueryImage};
use visloc_rs::localization::{
    BinaryLandmarkIndex, BinaryMatchGates, FixOutcome, MapAnchor, MapAnchorConfig,
};
use visloc_rs::vision::matching::HammingMatcher;
use visloc_rs::vision::pnp::{GaussNewtonPoseRefiner, P3PGrunert};
use visloc_rs::{
    AllLandmarksSelector, LocalizationConfig, LocalizationPipeline, PnPRansac,
    RadiusLandmarkSelector,
};

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
    /// Map landmarks closer than this count as one point in the ratio test.
    duplicate_radius_m: Option<f64>,
    /// Run the Basalt VIO alongside and anchor it to the map.
    vio: bool,
    calibration: Option<PathBuf>,
    config: Option<PathBuf>,
    /// With `--vio`, localize every N-th frame.
    localize_every: usize,
    /// Write each accepted localization's inliers here.
    dump_inliers: Option<PathBuf>,
}

impl Args {
    fn usage() -> String {
        "usage: basalt_euroc_localize_demo --map DIR --euroc-dir DIR [--out-dir DIR] \
         [--camera N] [--max-frames N] [--stride N] [--max-points N] [--max-hamming N] \
         [--ratio R|none] [--min-inliers N] [--reprojection-px PX] [--ransac-iterations N] \
         [--prior-radius-m M] [--duplicate-radius-m M] \
         [--vio --calibration FILE --config FILE [--localize-every N]] [--dump-inliers FILE]"
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
            duplicate_radius_m: None,
            vio: false,
            calibration: None,
            config: None,
            localize_every: 1,
            dump_inliers: None,
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
                "--duplicate-radius-m" => {
                    args.duplicate_radius_m = Some(number(&option, value()?)?);
                }
                "--vio" => args.vio = true,
                "--calibration" => args.calibration = Some(PathBuf::from(value()?)),
                "--config" => args.config = Some(PathBuf::from(value()?)),
                "--localize-every" => args.localize_every = number(&option, value()?)?,
                "--dump-inliers" => args.dump_inliers = Some(PathBuf::from(value()?)),
                other => return Err(format!("unknown option {other}\n\n{}", Self::usage())),
            }
        }
        args.map = map.ok_or_else(|| format!("--map is required\n\n{}", Self::usage()))?;
        args.euroc_dir =
            euroc_dir.ok_or_else(|| format!("--euroc-dir is required\n\n{}", Self::usage()))?;
        if args.stride == 0 {
            return Err("--stride must be positive".into());
        }
        if args.vio {
            if args.calibration.is_none() || args.config.is_none() {
                return Err("--vio needs --calibration and --config".into());
            }
            if args.stride != 1 {
                return Err("--vio runs on every frame; use --localize-every, not --stride".into());
            }
            if args.localize_every == 0 {
                return Err("--localize-every must be positive".into());
            }
        } else if args.localize_every != 1 || args.calibration.is_some() || args.config.is_some() {
            return Err("--localize-every, --calibration and --config need --vio".into());
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

/// One localization attempt on one image.
struct Attempt {
    result: LocalizationResult,
    used_prior: bool,
    keypoints: usize,
    latency_ms: f64,
    /// The localized IMU pose in the map frame, when accepted.
    world_from_imu: Option<SE3>,
    /// When accepted: each inlier's query keypoint and map landmark.
    inliers: Vec<(Point2<f64>, u64)>,
}

/// The saved map plus everything needed to localize one image against it.
struct Localizer {
    map: map_files::SavedMap,
    camera: Camera,
    camera_from_imu: SE3,
    max_points: usize,
    pipeline: Pipeline,
    /// The map's descriptors packed for the map-wide search, which gives
    /// the pipeline's correspondences (tested in `visloc-localization`).
    index: BinaryLandmarkIndex,
    gates: BinaryMatchGates,
    prior_radius_m: Option<f64>,
    previous: Option<Pose>,
}

impl Localizer {
    fn load(args: &Args) -> Result<Self, Box<dyn std::error::Error>> {
        let map = map_files::load(&args.map)?;
        let camera = map
            .map
            .cameras
            .get(&localization_camera_id(args.camera))
            .cloned()
            .ok_or_else(|| format!("camera {} is not in the map", args.camera))?;
        let imu_from_camera = map
            .imu_from_camera
            .get(usize::from(args.camera))
            .cloned()
            .ok_or_else(|| format!("map.json has no extrinsics for camera {}", args.camera))?;
        eprintln!(
            "map: images={} landmarks={} descriptors={}",
            map.map.keyframes.len(),
            map.map.landmarks.len(),
            map.descriptors.len()
        );
        let index = BinaryLandmarkIndex::new(&map.map, &map.descriptors)
            .with_duplicate_radius(args.duplicate_radius_m.unwrap_or(0.0));
        Ok(Self {
            max_points: args.max_points.unwrap_or(map.max_points),
            map,
            camera,
            camera_from_imu: imu_from_camera.inverse(),
            pipeline: pipeline(args),
            index,
            gates: BinaryMatchGates {
                max_distance: args.max_hamming,
                ratio: args.ratio,
            },
            prior_radius_m: args.prior_radius_m,
            previous: None,
        })
    }

    fn localize(&mut self, image: &RawU16Image) -> Result<Attempt, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let query = query_image(image, &self.camera, self.max_points)?;
        let prior_result = self
            .previous
            .as_ref()
            .zip(self.prior_radius_m)
            .map(|(pose, radius)| {
                self.pipeline
                    .localize_with_candidate_selector_and_descriptor_store(
                        &query,
                        &self.map.map,
                        &self.map.descriptors,
                        RadiusLandmarkSelector::new(pose.camera_center_world(), radius),
                    )
            });
        let used_prior = prior_result.as_ref().is_some_and(accepted);
        let result = match prior_result {
            Some(result) if used_prior => result,
            _ => self.pipeline.localize_correspondences(
                &self.camera,
                self.index.match_all(
                    &query.keypoints,
                    &self.index.pack_query(&query.descriptors),
                    self.gates,
                    1,
                ),
            ),
        };
        let latency_ms = started.elapsed().as_secs_f64() * 1e3;
        let success = accepted(&result);
        self.previous = success.then(|| result.pose.clone()).flatten();
        let world_from_imu = self
            .previous
            .as_ref()
            .map(|pose| pose.camera_to_world().compose(&self.camera_from_imu));
        let inliers = if world_from_imu.is_some() {
            result
                .inlier_query_indices
                .iter()
                .zip(&result.inlier_landmark_ids)
                .filter_map(|(&index, &landmark)| Some((*query.keypoints.get(index)?, landmark)))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Attempt {
            inliers,
            keypoints: query.keypoints.len(),
            result,
            used_prior,
            latency_ms,
            world_from_imu,
        })
    }
}

const CSV_HEADER: &str =
    "timestamp_ns,success,used_prior,keypoints,matches,inliers,inlier_ratio,latency_ms,anchor\n";

fn csv_row(csv: &mut String, timestamp_ns: i64, attempt: &Attempt, anchor: &str) {
    let _ = writeln!(
        csv,
        "{},{},{},{},{},{},{:.4},{:.2},{anchor}",
        timestamp_ns,
        attempt.world_from_imu.is_some(),
        attempt.used_prior,
        attempt.keypoints,
        attempt.result.match_count,
        attempt.result.inlier_count,
        attempt.result.inlier_ratio,
        attempt.latency_ms
    );
}

const INLIERS_HEADER: &str = "timestamp_ns,x,y,landmark_id\n";

fn inlier_rows(rows: &mut String, timestamp_ns: i64, attempt: &Attempt) {
    for (pixel, landmark) in &attempt.inliers {
        let _ = writeln!(
            rows,
            "{timestamp_ns},{:.2},{:.2},{landmark}",
            pixel.x, pixel.y
        );
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut localizer = Localizer::load(args)?;
    if args.vio {
        return run_vio(args, &mut localizer);
    }
    let camera_dir = args
        .euroc_dir
        .join("mav0")
        .join(format!("cam{}", args.camera));
    let images = read_camera_manifest(&camera_dir)?
        .into_iter()
        .step_by(args.stride)
        .take(args.max_frames.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();

    let mut tum = String::from("# timestamp tx ty tz qx qy qz qw\n");
    let mut csv = String::from(CSV_HEADER);
    let mut inliers = String::from(INLIERS_HEADER);
    let mut latencies = Vec::with_capacity(images.len());
    let mut localized = 0_usize;
    for entry in &images {
        let attempt = localizer.localize(&read_camera_image(&entry.path)?)?;
        latencies.push(attempt.latency_ms);
        if let Some(world_from_imu) = &attempt.world_from_imu {
            tum.push_str(&tum_line(entry.timestamp_ns, world_from_imu));
            localized += 1;
        }
        csv_row(&mut csv, entry.timestamp_ns, &attempt, "-");
        inlier_rows(&mut inliers, entry.timestamp_ns, &attempt);
    }

    fs::create_dir_all(&args.out_dir)?;
    fs::write(args.out_dir.join("localization.tum"), tum)?;
    fs::write(args.out_dir.join("per_frame.csv"), csv)?;
    if let Some(path) = &args.dump_inliers {
        fs::write(path, inliers)?;
    }
    println!(
        "localized {localized}/{} images (median {:.1} ms/image) -> {}",
        images.len(),
        median(latencies),
        args.out_dir.display()
    );
    Ok(())
}

/// Runs the Basalt VIO on every frame and anchors it to the map with the
/// fixes from every `--localize-every`-th frame.
fn run_vio(args: &Args, localizer: &mut Localizer) -> Result<(), Box<dyn std::error::Error>> {
    let calibration = args.calibration.as_ref().expect("checked by Args::parse");
    let config = args.config.as_ref().expect("checked by Args::parse");
    let dataset = EurocSensorDataset::open(&args.euroc_dir, calibration, config)?;
    let mut adapter =
        BasaltVioEstimatorAdapter::from_config(dataset.calibration(), dataset.config())?;
    adapter.estimator.set_lean_marg_data(true);
    let mut anchor = MapAnchor::new(MapAnchorConfig::default());
    let frames = args
        .max_frames
        .unwrap_or(usize::MAX)
        .min(dataset.frame_count());

    let header = "# timestamp tx ty tz qx qy qz qw\n";
    let (mut fixes_tum, mut map_tum, mut vio_tum) =
        (header.to_owned(), header.to_owned(), header.to_owned());
    let mut csv = String::from(CSV_HEADER);
    let mut inliers = String::from(INLIERS_HEADER);
    let mut map_csv =
        String::from("timestamp_ns,tx,ty,tz,qx,qy,qz,qw,seconds_since_fix,metres_since_fix\n");
    let mut latencies = Vec::new();
    let (mut attempts, mut localized, mut anchored) = (0_usize, 0_usize, 0_usize);
    let mut outcomes = std::collections::BTreeMap::<&str, usize>::new();
    let started = Instant::now();
    for index in 0..frames {
        let frame = dataset.frame(index)?;
        let timestamp_ns = frame.timestamp_ns;
        let attempt = if index % args.localize_every == 0 {
            let image = match args.camera {
                0 => Some(&frame.cam0),
                _ => frame.cam1.as_ref(),
            }
            .ok_or("this frame has no image for --camera")?;
            Some(localizer.localize(image)?)
        } else {
            None
        };
        let odom_from_imu = adapter
            .process_without_marg_data_no_trace(frame)?
            .estimator
            .state
            .imu_to_world;
        vio_tum.push_str(&tum_line(timestamp_ns, &odom_from_imu));

        if let Some(attempt) = attempt {
            attempts += 1;
            latencies.push(attempt.latency_ms);
            let outcome = match &attempt.world_from_imu {
                Some(map_from_imu) => {
                    localized += 1;
                    fixes_tum.push_str(&tum_line(timestamp_ns, map_from_imu));
                    match anchor.add_fix(timestamp_ns, map_from_imu, &odom_from_imu) {
                        FixOutcome::Pending => "pending",
                        FixOutcome::Initialized => "initialized",
                        FixOutcome::Accepted => "accepted",
                        FixOutcome::Rejected => "rejected",
                        FixOutcome::Reanchored => "reanchored",
                    }
                }
                None => "-",
            };
            *outcomes.entry(outcome).or_default() += 1;
            csv_row(&mut csv, timestamp_ns, &attempt, outcome);
            inlier_rows(&mut inliers, timestamp_ns, &attempt);
        }
        if let (Some(map_from_imu), Some((fix_ns, fix_position))) =
            (anchor.map_from_body(&odom_from_imu), anchor.last_fix())
        {
            map_tum.push_str(&tum_line(timestamp_ns, &map_from_imu));
            let (t, q) = (map_from_imu.translation, map_from_imu.rotation.quaternion());
            writeln!(
                map_csv,
                "{timestamp_ns},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.3},{:.3}",
                t.x,
                t.y,
                t.z,
                q.i,
                q.j,
                q.k,
                q.w,
                (timestamp_ns - fix_ns) as f64 * 1e-9,
                (odom_from_imu.translation - fix_position).norm(),
            )?;
            anchored += 1;
        }
    }
    let wall_s = started.elapsed().as_secs_f64();

    fs::create_dir_all(&args.out_dir)?;
    fs::write(args.out_dir.join("localization.tum"), fixes_tum)?;
    fs::write(args.out_dir.join("trajectory_map.tum"), map_tum)?;
    fs::write(args.out_dir.join("trajectory_map.csv"), map_csv)?;
    fs::write(args.out_dir.join("trajectory_vio.tum"), vio_tum)?;
    fs::write(args.out_dir.join("per_frame.csv"), csv)?;
    if let Some(path) = &args.dump_inliers {
        fs::write(path, inliers)?;
    }
    println!(
        "frames {frames}, localized {localized}/{attempts} attempts (median {:.1} ms), \
         anchored {anchored}/{frames} frames, fixes {outcomes:?}, {wall_s:.1} s -> {}",
        median(latencies),
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

    #[test]
    fn parser_checks_the_vio_options() {
        let base = ["--map", "m", "--euroc-dir", "d"];
        let with = |extra: &[&str]| parse(&base.iter().chain(extra).copied().collect::<Vec<_>>());
        assert!(with(&["--vio"]).is_err());
        assert!(with(&["--vio", "--calibration", "c.json"]).is_err());
        assert!(with(&["--localize-every", "5"]).is_err());
        assert!(with(&["--calibration", "c.json"]).is_err());
        let vio = ["--vio", "--calibration", "c.json", "--config", "k.json"];
        assert!(with(&[&vio[..], &["--stride", "2"]].concat()).is_err());
        assert!(with(&[&vio[..], &["--localize-every", "0"]].concat()).is_err());
        let args = with(&[&vio[..], &["--localize-every", "5"]].concat()).expect("parses");
        assert!(args.vio);
        assert_eq!(args.localize_every, 5);
        assert_eq!(args.config, Some(PathBuf::from("k.json")));
        assert_eq!(with(&[]).expect("parses").localize_every, 1);
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
