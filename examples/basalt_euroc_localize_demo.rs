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
//! the whole map as the fallback. With `--retrieval-keyframes N`, the
//! map-wide search first matches only among the landmarks of the `N` map
//! keyframes that look most like the image
//! (`visloc_localization::KeyframeRetrieval`, `--retrieval-bits`-bit hashed
//! words, 16 by default), and falls back to the whole map when that does
//! not localize.
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
//! localized; until the correction is set, every `--initial-localize-every`-th
//! (1: every image), as two agreeing fixes are needed before any frame gets
//! a map pose. That holds within `--initial-window-s` (30) seconds of the
//! start or of the latest localization; after a longer stretch without one
//! the map likely does not cover the place, and every `--localize-every`-th
//! image is tried until one localizes again. Accepted fixes correct the VIO into the map frame
//! (`visloc_localization::OdomCorrection`: a gravity-aligned yaw +
//! translation offset, gated against outliers), so every frame gets a
//! map-frame pose once the correction is set, including the frames the map
//! does not localize. From then on the corrected VIO predicts each image's
//! pose and every landmark is first matched only against the keypoints within
//! `--projection-radius-px` (20) pixels of its predicted projection, as
//! `visloc_map_localize_node` does; the map-wide search remains the fallback
//! (`--projection-radius-px 0` matches map-wide only).
//! The prediction is used only within `--projection-fresh-s` (5) seconds
//! and `--projection-fresh-m` (2) metres of odometry travel after the
//! correction's newest fix: a drifted prediction can steer the projection search
//! to a wrong pose that agrees with it.
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
//!   latency, with `--vio` what the correction did with the fix, and which
//!   search gave the result (`projection`, `radius`, `retrieval` or
//!   `global`).
//! - With `--vio`: `trajectory_map.tum`, the IMU pose in the map frame of
//!   every frame from the correction's start; `trajectory_map.csv`, the same
//!   poses with the time and VIO distance since the correction's newest fix
//!   (the fused error grows with
//!   them, so use them to judge a pose); `trajectory_vio.tum`, the raw VIO in
//!   its own frame.
//! - With `--odometry FILE` instead of `--vio`: `trajectory_map.tum` and
//!   `.csv` as above, and `trajectory_odometry.tum`, the odometry at the
//!   image stamps. `FILE` is a TUM trajectory of a body (e.g. a wheeled
//!   robot's `base_link` in its odometry frame), interpolated at each image
//!   stamp; `--imu-from-body tx,ty,tz,qx,qy,qz,qw` places the body in the
//!   map's IMU frame. The odometry frame must be gravity-aligned (z up), as
//!   planar wheel odometry on a level floor is.
//! - With `--dump-inliers FILE`: per accepted localization, one CSV row per
//!   PnP inlier, `timestamp_ns,x,y,landmark_id` (the query keypoint in
//!   pixels and the map landmark it matched), e.g. for drawing the matches.

#[path = "basalt_map/map_files.rs"]
mod map_files;

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{Point2, Quaternion, UnitQuaternion, Vector3};
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
    BinaryLandmarkIndex, BinaryMatchGates, FixOutcome, KeyframeRetrieval, OdomCorrection,
    OdomCorrectionConfig,
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
    /// Match map-wide among the landmarks of the N retrieved keyframes first.
    retrieval_keyframes: Option<usize>,
    /// Bits of a retrieval visual word.
    retrieval_bits: u8,
    /// Run the Basalt VIO alongside and correct it into the map frame.
    vio: bool,
    calibration: Option<PathBuf>,
    config: Option<PathBuf>,
    /// Correct this odometry (TUM, a body's poses) instead of the VIO.
    odometry: Option<PathBuf>,
    /// With `--odometry`: the body's pose in the IMU frame, `T_imu_body`.
    imu_from_body: Option<SE3>,
    /// With `--vio` or `--odometry`, localize every N-th frame.
    localize_every: usize,
    /// ... and every N-th until the odometry has a map pose,
    initial_localize_every: usize,
    /// ... within this many seconds of the start or of the latest
    /// localization.
    initial_window_s: f64,
    /// With `--vio` or `--odometry`, match by projection from the corrected
    /// odometry's prediction first, within this radius (20 px unless set;
    /// `0` turns it off).
    projection_radius_px: Option<f64>,
    /// Trust the prediction only this long and this far after the
    /// correction's newest fix.
    projection_fresh_s: f64,
    projection_fresh_m: f64,
    /// Write each accepted localization's inliers here.
    dump_inliers: Option<PathBuf>,
}

impl Args {
    fn usage() -> String {
        "usage: basalt_euroc_localize_demo --map DIR --euroc-dir DIR [--out-dir DIR] \
         [--camera N] [--max-frames N] [--stride N] [--max-points N] [--max-hamming N] \
         [--ratio R|none] [--min-inliers N] [--reprojection-px PX] [--ransac-iterations N] \
         [--prior-radius-m M] [--duplicate-radius-m M] \
         [--retrieval-keyframes N [--retrieval-bits B]] \
         [--vio --calibration FILE --config FILE | --odometry FILE \
         [--imu-from-body tx,ty,tz,qx,qy,qz,qw]] [--localize-every N] \
         [--initial-localize-every N] [--initial-window-s S] \
         [--projection-radius-px PX [--projection-fresh-s S] [--projection-fresh-m M]] \
         [--dump-inliers FILE]"
            .into()
    }

    fn parse<I>(arguments: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut map = None;
        let mut euroc_dir = None;
        let mut projection_radius_px = None::<f64>;
        let mut initial_window_s = None::<f64>;
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
            retrieval_keyframes: None,
            retrieval_bits: 16,
            vio: false,
            calibration: None,
            config: None,
            odometry: None,
            imu_from_body: None,
            localize_every: 1,
            initial_localize_every: 1,
            initial_window_s: 30.0,
            projection_radius_px: None,
            projection_fresh_s: 5.0,
            projection_fresh_m: 2.0,
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
                "--retrieval-keyframes" => {
                    args.retrieval_keyframes = Some(number(&option, value()?)?);
                }
                "--retrieval-bits" => args.retrieval_bits = number(&option, value()?)?,
                "--vio" => args.vio = true,
                "--calibration" => args.calibration = Some(PathBuf::from(value()?)),
                "--config" => args.config = Some(PathBuf::from(value()?)),
                "--odometry" => args.odometry = Some(PathBuf::from(value()?)),
                "--imu-from-body" => {
                    let value = value()?;
                    let numbers = value
                        .split(',')
                        .map(|field| number::<f64>(&option, field.trim().to_owned()))
                        .collect::<Result<Vec<_>, _>>()?;
                    let [tx, ty, tz, qx, qy, qz, qw] = numbers[..] else {
                        return Err(format!("{option}: expected tx,ty,tz,qx,qy,qz,qw"));
                    };
                    args.imu_from_body = Some(SE3::new(
                        UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
                        Vector3::new(tx, ty, tz),
                    ));
                }
                "--localize-every" => args.localize_every = number(&option, value()?)?,
                "--initial-localize-every" => {
                    args.initial_localize_every = number(&option, value()?)?;
                }
                "--initial-window-s" => initial_window_s = Some(number(&option, value()?)?),
                "--projection-radius-px" => {
                    projection_radius_px = Some(number(&option, value()?)?);
                }
                "--projection-fresh-s" => args.projection_fresh_s = number(&option, value()?)?,
                "--projection-fresh-m" => args.projection_fresh_m = number(&option, value()?)?,
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
        if args.retrieval_keyframes == Some(0) || !(1..=32).contains(&args.retrieval_bits) {
            return Err("--retrieval-keyframes must be positive, --retrieval-bits 1..=32".into());
        }
        if args.vio && args.odometry.is_some() {
            return Err("--vio and --odometry are alternatives".into());
        }
        if args.vio != (args.calibration.is_some() && args.config.is_some())
            || args.calibration.is_some() != args.config.is_some()
        {
            return Err("--vio needs --calibration and --config, which need --vio".into());
        }
        if args.imu_from_body.is_some() && args.odometry.is_none() {
            return Err("--imu-from-body needs --odometry".into());
        }
        if args.vio || args.odometry.is_some() {
            if args.stride != 1 {
                return Err(
                    "a --vio or --odometry run sees every frame; use --localize-every, not --stride"
                        .into(),
                );
            }
            if args.localize_every == 0 || args.initial_localize_every == 0 {
                return Err(
                    "--localize-every and --initial-localize-every must be positive".into(),
                );
            }
            if let Some(window) = initial_window_s {
                if !(window >= 0.0 && window.is_finite()) {
                    return Err("--initial-window-s must be a non-negative number".into());
                }
                args.initial_window_s = window;
            }
        } else if args.localize_every != 1
            || args.initial_localize_every != 1
            || initial_window_s.is_some()
            || projection_radius_px.is_some()
        {
            return Err(
                "--localize-every, --initial-localize-every, --initial-window-s and \
                 --projection-radius-px need --vio or --odometry"
                    .into(),
            );
        }
        args.projection_radius_px = match projection_radius_px {
            Some(radius) => (radius > 0.0).then_some(radius),
            None => (args.vio || args.odometry.is_some()).then_some(20.0),
        };
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
    /// The search that gave `result`: `projection`, `radius`, `retrieval`
    /// or `global`.
    matching: &'static str,
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
    projection_radius_px: Option<f64>,
    /// Keyframe retrieval, and how many keyframes' landmarks to match among.
    retrieval: Option<(KeyframeRetrieval, usize)>,
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
        let retrieval = args.retrieval_keyframes.map(|top_k| {
            (
                KeyframeRetrieval::new(&map.map, &index, args.retrieval_bits),
                top_k,
            )
        });
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
            projection_radius_px: args.projection_radius_px,
            retrieval,
            previous: None,
        })
    }

    /// Localizes one image. `predicted`, the map-frame IMU pose the corrected
    /// VIO predicts for it, enables the projection search.
    fn localize(
        &mut self,
        image: &RawU16Image,
        predicted: Option<&SE3>,
    ) -> Result<Attempt, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let query = query_image(image, &self.camera, self.max_points)?;
        let packed = self.index.pack_query(&query.descriptors);
        let projected = predicted
            .zip(self.projection_radius_px)
            .map(|(map_from_imu, radius)| {
                let prior = Pose {
                    world_to_camera: self.camera_from_imu.compose(&map_from_imu.inverse()),
                };
                self.pipeline.localize_correspondences(
                    &self.camera,
                    self.index.match_projected(
                        &self.camera,
                        &prior,
                        &query.keypoints,
                        &packed,
                        radius,
                        self.gates,
                    ),
                )
            })
            .filter(accepted);
        let prior_result = projected.or_else(|| {
            self.previous
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
                })
        });
        let used_prior = prior_result.as_ref().is_some_and(accepted);
        let mut matching = if !used_prior {
            "global"
        } else if predicted.is_some() && self.projection_radius_px.is_some() {
            "projection"
        } else {
            "radius"
        };
        let result = match prior_result {
            Some(result) if used_prior => result,
            _ => {
                // The retrieved keyframes' landmarks first, then the whole map.
                let retrieved = self
                    .retrieval
                    .as_ref()
                    .map(|(retrieval, top_k)| {
                        let landmarks = retrieval.candidate_landmarks(&packed, *top_k);
                        self.pipeline.localize_correspondences(
                            &self.camera,
                            self.index.match_among(
                                &query.keypoints,
                                &packed,
                                self.gates,
                                &landmarks,
                            ),
                        )
                    })
                    .filter(accepted);
                match retrieved {
                    Some(result) => {
                        matching = "retrieval";
                        result
                    }
                    None => self.pipeline.localize_correspondences(
                        &self.camera,
                        self.index
                            .match_all(&query.keypoints, &packed, self.gates, 1),
                    ),
                }
            }
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
            matching,
            latency_ms,
            world_from_imu,
        })
    }
}

const CSV_HEADER: &str =
    "timestamp_ns,success,used_prior,keypoints,matches,inliers,inlier_ratio,latency_ms,correction,\
     matching\n";

fn csv_row(csv: &mut String, timestamp_ns: i64, attempt: &Attempt, correction: &str) {
    let _ = writeln!(
        csv,
        "{},{},{},{},{},{},{:.4},{:.2},{correction},{}",
        timestamp_ns,
        attempt.world_from_imu.is_some(),
        attempt.used_prior,
        attempt.keypoints,
        attempt.result.match_count,
        attempt.result.inlier_count,
        attempt.result.inlier_ratio,
        attempt.latency_ms,
        attempt.matching
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
    if let Some(path) = &args.odometry {
        return run_odometry(args, &mut localizer, path);
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
        let attempt = localizer.localize(&read_camera_image(&entry.path)?, None)?;
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

/// Odometry corrected into the map frame, frame by frame: the fixes, the
/// correction and every output file of `--vio` / `--odometry`.
struct CorrectedRun {
    correction: OdomCorrection,
    /// Localize every `localize_every`-th frame once the odometry has a map
    /// pose, every `initial_every`-th before, within `initial_window_ns` of
    /// the first frame or of the latest localization.
    localize_every: usize,
    initial_every: usize,
    initial_window_ns: i64,
    first_ns: Option<i64>,
    last_localized_ns: Option<i64>,
    /// How long / far after the correction's newest fix its prediction is
    /// used.
    fresh_s: f64,
    fresh_m: f64,
    fixes_tum: String,
    map_tum: String,
    odometry_tum: String,
    csv: String,
    inliers: String,
    map_csv: String,
    latencies: Vec<f64>,
    attempts: usize,
    localized: usize,
    map_poses: usize,
    frames: usize,
    outcomes: std::collections::BTreeMap<&'static str, usize>,
    started: Instant,
}

impl CorrectedRun {
    fn new(args: &Args) -> Self {
        let header = "# timestamp tx ty tz qx qy qz qw\n";
        Self {
            correction: OdomCorrection::new(OdomCorrectionConfig::default()),
            localize_every: args.localize_every,
            initial_every: args.initial_localize_every,
            initial_window_ns: (args.initial_window_s * 1e9) as i64,
            first_ns: None,
            last_localized_ns: None,
            fresh_s: args.projection_fresh_s,
            fresh_m: args.projection_fresh_m,
            fixes_tum: header.to_owned(),
            map_tum: header.to_owned(),
            odometry_tum: header.to_owned(),
            csv: String::from(CSV_HEADER),
            inliers: String::from(INLIERS_HEADER),
            map_csv: String::from(
                "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,seconds_since_fix,metres_since_fix\n",
            ),
            latencies: Vec::new(),
            attempts: 0,
            localized: 0,
            map_poses: 0,
            frames: 0,
            outcomes: std::collections::BTreeMap::new(),
            started: Instant::now(),
        }
    }

    /// Whether to localize the `index`-th frame, stamped `timestamp_ns`.
    /// Until the correction is set, its first fixes are what gives the
    /// odometry a map pose at all, so every `initial_every`-th frame is
    /// tried. After `initial_window_ns` without a localization (from the
    /// first frame, or the latest localization), the map likely does not
    /// cover the place, and every `localize_every`-th is tried until one
    /// localizes; once the correction is set, every `localize_every`-th.
    fn localizes(&self, index: usize, timestamp_ns: i64) -> bool {
        let since = self
            .last_localized_ns
            .or(self.first_ns)
            .map_or(0, |start_ns| timestamp_ns - start_ns);
        let every = if self.correction.is_set() || since > self.initial_window_ns {
            self.localize_every
        } else {
            self.initial_every
        };
        index.is_multiple_of(every)
    }

    /// One frame: the odometry's IMU pose and, on a frame to localize, its
    /// image.
    fn frame(
        &mut self,
        localizer: &mut Localizer,
        timestamp_ns: i64,
        odom_from_imu: &SE3,
        image: Option<&RawU16Image>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.frames += 1;
        self.first_ns.get_or_insert(timestamp_ns);
        self.odometry_tum
            .push_str(&tum_line(timestamp_ns, odom_from_imu));
        // The correction as it stands before this image's fix predicts its pose,
        // while its newest fix is recent: after a long gap, a drifted
        // prediction can steer the projection search to a wrong pose that
        // agrees with it.
        let predicted = self
            .correction
            .last_fix()
            .filter(|(fix_ns, fix_position)| {
                (timestamp_ns - fix_ns) as f64 * 1e-9 <= self.fresh_s
                    && (odom_from_imu.translation - fix_position).norm() <= self.fresh_m
            })
            .and_then(|_| self.correction.map_from_body(odom_from_imu));
        if let Some(image) = image {
            let attempt = localizer.localize(image, predicted.as_ref())?;
            self.attempts += 1;
            self.latencies.push(attempt.latency_ms);
            let outcome = match &attempt.world_from_imu {
                Some(map_from_imu) => {
                    self.localized += 1;
                    self.last_localized_ns = Some(timestamp_ns);
                    self.fixes_tum
                        .push_str(&tum_line(timestamp_ns, map_from_imu));
                    match self
                        .correction
                        .add_fix(timestamp_ns, map_from_imu, odom_from_imu)
                    {
                        FixOutcome::Pending => "pending",
                        FixOutcome::Initialized => "initialized",
                        FixOutcome::Accepted => "accepted",
                        FixOutcome::Rejected => "rejected",
                        FixOutcome::Reset => "reset",
                    }
                }
                None => "-",
            };
            *self.outcomes.entry(outcome).or_default() += 1;
            csv_row(&mut self.csv, timestamp_ns, &attempt, outcome);
            inlier_rows(&mut self.inliers, timestamp_ns, &attempt);
        }
        if let (Some(map_from_imu), Some((fix_ns, fix_position))) = (
            self.correction.map_from_body(odom_from_imu),
            self.correction.last_fix(),
        ) {
            self.map_tum
                .push_str(&tum_line(timestamp_ns, &map_from_imu));
            let (t, q) = (map_from_imu.translation, map_from_imu.rotation.quaternion());
            writeln!(
                self.map_csv,
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
            self.map_poses += 1;
        }
        Ok(())
    }

    /// Writes the outputs; the odometry's own trajectory goes to
    /// `odometry_file`.
    fn finish(self, args: &Args, odometry_file: &str) -> Result<(), Box<dyn std::error::Error>> {
        let wall_s = self.started.elapsed().as_secs_f64();
        fs::create_dir_all(&args.out_dir)?;
        fs::write(args.out_dir.join("localization.tum"), self.fixes_tum)?;
        fs::write(args.out_dir.join("trajectory_map.tum"), self.map_tum)?;
        fs::write(args.out_dir.join("trajectory_map.csv"), self.map_csv)?;
        fs::write(args.out_dir.join(odometry_file), self.odometry_tum)?;
        fs::write(args.out_dir.join("per_frame.csv"), self.csv)?;
        if let Some(path) = &args.dump_inliers {
            fs::write(path, self.inliers)?;
        }
        println!(
            "frames {}, localized {}/{} attempts (median {:.1} ms), map poses {}/{} frames, \
             fixes {:?}, {wall_s:.1} s -> {}",
            self.frames,
            self.localized,
            self.attempts,
            median(self.latencies),
            self.map_poses,
            self.frames,
            self.outcomes,
            args.out_dir.display()
        );
        Ok(())
    }
}

/// Runs the Basalt VIO on every frame and corrects it into the map frame with
/// the fixes from every `--localize-every`-th frame (every
/// `--initial-localize-every`-th until the odometry has a map pose).
fn run_vio(args: &Args, localizer: &mut Localizer) -> Result<(), Box<dyn std::error::Error>> {
    let calibration = args.calibration.as_ref().expect("checked by Args::parse");
    let config = args.config.as_ref().expect("checked by Args::parse");
    let dataset = EurocSensorDataset::open(&args.euroc_dir, calibration, config)?;
    let mut adapter =
        BasaltVioEstimatorAdapter::from_config(dataset.calibration(), dataset.config())?;
    adapter.estimator.set_lean_marg_data(true);
    let frames = args
        .max_frames
        .unwrap_or(usize::MAX)
        .min(dataset.frame_count());

    let mut run = CorrectedRun::new(args);
    for index in 0..frames {
        let frame = dataset.frame(index)?;
        let timestamp_ns = frame.timestamp_ns;
        let image = if run.localizes(index, timestamp_ns) {
            let image = match args.camera {
                0 => Some(&frame.cam0),
                _ => frame.cam1.as_ref(),
            }
            .ok_or("this frame has no image for --camera")?;
            Some(image.clone())
        } else {
            None
        };
        let odom_from_imu = adapter
            .process_without_marg_data_no_trace(frame)?
            .estimator
            .state
            .imu_to_world;
        run.frame(localizer, timestamp_ns, &odom_from_imu, image.as_ref())?;
    }
    run.finish(args, "trajectory_vio.tum")
}

/// Corrects an external odometry (`--odometry`, e.g. a robot's wheel
/// odometry) into the map frame instead of the VIO: each camera image takes the
/// odometry pose interpolated at its stamp, and every `--localize-every`-th
/// image is localized (every `--initial-localize-every`-th until the odometry
/// has a map pose).
fn run_odometry(
    args: &Args,
    localizer: &mut Localizer,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let odometry = read_tum_ns(path)?;
    let body_from_imu = args
        .imu_from_body
        .as_ref()
        .map_or_else(SE3::identity, SE3::inverse);
    let camera_dir = args
        .euroc_dir
        .join("mav0")
        .join(format!("cam{}", args.camera));
    let images = read_camera_manifest(&camera_dir)?
        .into_iter()
        .take(args.max_frames.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();

    let mut run = CorrectedRun::new(args);
    for (index, entry) in images.iter().enumerate() {
        let Some(odom_from_body) = interpolate_pose(&odometry, entry.timestamp_ns) else {
            continue;
        };
        let odom_from_imu = odom_from_body.compose(&body_from_imu);
        let image = run
            .localizes(index, entry.timestamp_ns)
            .then(|| read_camera_image(&entry.path))
            .transpose()?;
        run.frame(
            localizer,
            entry.timestamp_ns,
            &odom_from_imu,
            image.as_ref(),
        )?;
    }
    run.finish(args, "trajectory_odometry.tum")
}

/// A TUM trajectory (`timestamp tx ty tz qx qy qz qw`, seconds) with exact
/// nanosecond stamps.
fn read_tum_ns(path: &std::path::Path) -> Result<Vec<(i64, SE3)>, Box<dyn std::error::Error>> {
    let mut poses = Vec::new();
    for (line_index, line) in fs::read_to_string(path)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let bad = || format!("{}:{}: expected a TUM pose", path.display(), line_index + 1);
        if fields.len() < 8 {
            return Err(bad().into());
        }
        let (seconds, fraction) = fields[0].split_once('.').unwrap_or((fields[0], ""));
        let fraction = format!("{fraction:0<9}");
        let timestamp_ns = seconds.parse::<i64>().map_err(|_| bad())? * 1_000_000_000
            + fraction[..9].parse::<i64>().map_err(|_| bad())?;
        let values = fields[1..8]
            .iter()
            .map(|field| field.parse::<f64>().map_err(|_| bad()))
            .collect::<Result<Vec<_>, _>>()?;
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
            values[6], values[3], values[4], values[5],
        ));
        poses.push((
            timestamp_ns,
            SE3::new(rotation, Vector3::new(values[0], values[1], values[2])),
        ));
    }
    poses.sort_by_key(|(timestamp_ns, _)| *timestamp_ns);
    Ok(poses)
}

/// The pose at `timestamp_ns`, interpolated between the two poses around
/// it (linear translation, slerp rotation); `None` outside the trajectory.
fn interpolate_pose(poses: &[(i64, SE3)], timestamp_ns: i64) -> Option<SE3> {
    let after = poses.partition_point(|(t, _)| *t < timestamp_ns);
    let (t1, pose1) = poses.get(after)?;
    if *t1 == timestamp_ns {
        return Some(pose1.clone());
    }
    let (t0, pose0) = poses.get(after.checked_sub(1)?)?;
    let s = (timestamp_ns - t0) as f64 / (t1 - t0) as f64;
    Some(SE3::new(
        pose0.rotation.slerp(&pose1.rotation, s),
        pose0.translation + (pose1.translation - pose0.translation) * s,
    ))
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
        assert!(with(&["--projection-radius-px", "20"]).is_err());
        let vio = ["--vio", "--calibration", "c.json", "--config", "k.json"];
        assert!(with(&[&vio[..], &["--stride", "2"]].concat()).is_err());
        assert!(with(&[&vio[..], &["--localize-every", "0"]].concat()).is_err());
        assert!(with(&["--initial-localize-every", "1"]).is_ok());
        assert!(with(&["--initial-localize-every", "5"]).is_err());
        assert!(with(&[&vio[..], &["--initial-localize-every", "0"]].concat()).is_err());
        assert!(with(&["--initial-window-s", "30"]).is_err());
        assert!(with(&[&vio[..], &["--initial-window-s", "-1"]].concat()).is_err());
        assert!(with(&[&vio[..], &["--initial-window-s", "inf"]].concat()).is_err());
        let args = with(&[&vio[..], &["--localize-every", "5"]].concat()).expect("parses");
        assert!(args.vio);
        assert_eq!(args.localize_every, 5);
        assert_eq!(args.initial_localize_every, 1);
        let args = with(&[&vio[..], &["--initial-localize-every", "5"]].concat()).expect("parses");
        assert_eq!(args.initial_localize_every, 5);
        assert_eq!(args.initial_window_s, 30.0);
        let args = with(&[&vio[..], &["--initial-window-s", "0"]].concat()).expect("parses");
        assert_eq!(args.initial_window_s, 0.0);
        assert_eq!(args.config, Some(PathBuf::from("k.json")));
        assert_eq!(args.projection_radius_px, Some(20.0));
        let args = with(&[&vio[..], &["--projection-radius-px", "12"]].concat()).expect("parses");
        assert_eq!(args.projection_radius_px, Some(12.0));
        let args = with(&[&vio[..], &["--projection-radius-px", "0"]].concat()).expect("parses");
        assert_eq!(args.projection_radius_px, None);
        assert_eq!(with(&[]).expect("parses").projection_radius_px, None);
        assert_eq!(with(&[]).expect("parses").localize_every, 1);
    }

    #[test]
    fn parser_checks_the_odometry_options() {
        let base = ["--map", "m", "--euroc-dir", "d"];
        let with = |extra: &[&str]| parse(&base.iter().chain(extra).copied().collect::<Vec<_>>());
        let args = with(&[
            "--odometry",
            "odom.tum",
            "--imu-from-body",
            "0.1,0,0.3,0,0,0.7071068,0.7071068",
            "--localize-every",
            "5",
            "--projection-radius-px",
            "20",
        ])
        .expect("parses");
        assert_eq!(args.odometry, Some(PathBuf::from("odom.tum")));
        let imu_from_body = args.imu_from_body.expect("extrinsics");
        assert!((imu_from_body.translation - Vector3::new(0.1, 0.0, 0.3)).norm() < 1e-12);
        assert!((imu_from_body.rotation.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(args.localize_every, 5);
        assert!(with(&["--imu-from-body", "0,0,0,0,0,0,1"]).is_err());
        assert!(with(&["--odometry", "o.tum", "--imu-from-body", "1,2,3"]).is_err());
        assert!(with(&["--odometry", "o.tum", "--stride", "2"]).is_err());
        let vio = ["--vio", "--calibration", "c.json", "--config", "k.json"];
        assert!(with(&[&vio[..], &["--odometry", "o.tum"]].concat()).is_err());
    }

    #[test]
    fn every_image_is_localized_until_the_first_map_pose() {
        let args = parse(&[
            "--map",
            "m",
            "--euroc-dir",
            "d",
            "--odometry",
            "o.tum",
            "--localize-every",
            "5",
        ])
        .expect("parses");
        let mut run = CorrectedRun::new(&args);
        let s = |seconds: f64| (seconds * 1e9) as i64;
        assert!((0..7).all(|index| run.localizes(index, 0)));
        // Within 30 s of the first frame every image; after that every 5th
        // until an image localizes, then every image for 30 s again.
        run.first_ns = Some(s(10.0));
        assert!((0..7).all(|index| run.localizes(index, s(40.0))));
        assert_eq!(
            (0..12)
                .filter(|&index| run.localizes(index, s(40.1)))
                .collect::<Vec<_>>(),
            [0, 5, 10]
        );
        run.last_localized_ns = Some(s(55.0));
        assert!((0..7).all(|index| run.localizes(index, s(85.0))));
        assert!(!run.localizes(1, s(85.1)));
        // Two agreeing fixes set the correction.
        for (x, timestamp_ns) in [(0.0, 0), (0.1, 100_000_000)] {
            let pose = SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0));
            run.correction.add_fix(timestamp_ns, &pose, &pose);
        }
        assert!(run.correction.is_set());
        let localized = (0..12)
            .filter(|&index| run.localizes(index, s(56.0)))
            .collect::<Vec<_>>();
        assert_eq!(localized, [0, 5, 10]);
    }

    #[test]
    fn odometry_is_read_exactly_and_interpolated() {
        let path = std::env::temp_dir().join(format!("visloc_odometry_{}.tum", std::process::id()));
        fs::write(
            &path,
            "# timestamp tx ty tz qx qy qz qw\n\
             1560000110.95848536 1 0 0 0 0 0 1\n\
             1560000111.00848532 2 0 0 0 0 0.7071067811865476 0.7071067811865476\n",
        )
        .expect("write");
        let poses = read_tum_ns(&path).expect("reads");
        fs::remove_file(&path).expect("cleanup");
        assert_eq!(poses[0].0, 1_560_000_110_958_485_360);
        assert_eq!(poses[1].0, 1_560_000_111_008_485_320);
        let middle = (poses[0].0 + poses[1].0) / 2;
        let pose = interpolate_pose(&poses, middle).expect("inside");
        assert!((pose.translation.x - 1.5).abs() < 1e-6);
        assert!((pose.rotation.angle() - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
        assert!(interpolate_pose(&poses, poses[0].0 - 1).is_none());
        assert!(interpolate_pose(&poses, poses[1].0 + 1).is_none());
        assert_eq!(
            interpolate_pose(&poses, poses[1].0)
                .expect("end")
                .translation
                .x,
            2.0
        );
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
        // Each landmark's track is read back, so a map written again (as
        // basalt_map_merge does) keeps it.
        let track_length = |map: &visloc_core::types::VisualMap| {
            map.landmarks
                .values()
                .map(|landmark| landmark.observations.len())
                .sum::<usize>()
        };
        assert!(track_length(&export.map) > 0);
        assert_eq!(track_length(&saved.map), track_length(&export.map));
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
