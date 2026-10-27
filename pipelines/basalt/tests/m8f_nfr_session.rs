//! Focused release gate for the stateful `NfrMapper::addMargData` boundary.
//!
//! The packet is the real pinned MH_01 M8a/M8b fixture, not a generated
//! identity system.  The tests exercise packet ingestion and the first
//! stateful image-frontend handoff; AOM construction and IMU factor
//! generation remain outside their scope.

use std::collections::BTreeSet;

use nalgebra::{Point2, UnitQuaternion, Vector3};
use serde_json::{json, Value};
use visloc_core::geometry::SE3;

use visloc_basalt::{
    camera::DoubleSphereCamera,
    mapper::{
        GlobalBaConfig, MapperImageFeatures, MapperLandmark, MapperObservation, MatchData,
        NfrMapper, NfrMapperDetectionReport, NfrMapperHeadlessConfig, NfrMapperIngestReport,
        NfrMapperStereoReport, SetupOptRejectReason, TimeCamId,
    },
    vio::{
        landmarks::StereographicDirection,
        margdata::{
            AomBlockData, FramePoseData, FrameStateData, MargData, MarginalizationTargets,
            MatrixData, OfImageData, MARGDATA_SCHEMA_VERSION_V3,
        },
    },
    BasaltCalibration,
};

const FIXTURE: &str =
    include_str!("../../../benchmarks/basalt/m8a_m8b_mh01_1403636579763555584.json");

fn array(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|item| item.as_f64().expect("finite number"))
        .collect()
}

fn fixture_packet() -> MargData {
    let root: Value = serde_json::from_str(FIXTURE).expect("M8a/M8b fixture JSON");
    let input = &root["input"];

    let h_rows = input["abs_H"].as_array().expect("input abs_H");
    let h_cols = h_rows
        .first()
        .and_then(Value::as_array)
        .expect("abs_H row")
        .len();
    let mut h_column_major = Vec::with_capacity(h_cols * h_rows.len());
    for col in 0..h_cols {
        for row in h_rows {
            h_column_major.push(
                row.as_array()
                    .and_then(|values| values.get(col))
                    .and_then(Value::as_f64)
                    .expect("abs_H entry"),
            );
        }
    }

    let frame_poses = input["frame_poses"]
        .as_array()
        .expect("frame_poses")
        .iter()
        .map(|pose| {
            let q = array(&pose["pose"]["quaternion_xyzw"]);
            let t = array(&pose["pose"]["translation"]);
            FramePoseData {
                frame_id: pose["id"].as_u64().expect("pose id"),
                timestamp_ns: pose["t_ns"].as_i64().expect("pose timestamp"),
                pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
                is_keyframe: true,
            }
        })
        .collect::<Vec<_>>();

    let frame_states = input["frame_states"]
        .as_array()
        .expect("frame_states")
        .iter()
        .map(|state| {
            let q = array(&state["pose"]["quaternion_xyzw"]);
            let t = array(&state["pose"]["translation"]);
            let velocity = array(&state["velocity"]);
            let gyro_bias = array(&state["bias_gyro"]);
            let accel_bias = array(&state["bias_accel"]);
            FrameStateData {
                frame_id: state["id"].as_u64().expect("state id"),
                timestamp_ns: state["t_ns"].as_i64().expect("state timestamp"),
                pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
                velocity: velocity.try_into().expect("velocity length"),
                gyro_bias: gyro_bias.try_into().expect("gyro bias length"),
                accel_bias: accel_bias.try_into().expect("accel bias length"),
                linearized: state["linearized"].as_bool().expect("linearized"),
                is_keyframe: input["kfs_all"]
                    .as_array()
                    .expect("kfs_all")
                    .iter()
                    .any(|id| id.as_u64() == Some(state["id"].as_u64().unwrap())),
                is_latest: false,
            }
        })
        .collect::<Vec<_>>();

    let aom_order = input["aom"]["blocks"]
        .as_array()
        .expect("AOM blocks")
        .iter()
        .map(|block| AomBlockData {
            frame_id: block["id"].as_u64().expect("block id"),
            offset: block["offset"].as_u64().expect("block offset") as usize,
            dof: block["size"].as_u64().expect("block size") as usize,
            kind: if block["size"].as_u64() == Some(6) {
                "pose".into()
            } else {
                "state".into()
            },
        })
        .collect::<Vec<_>>();

    MargData {
        // Pinned M8f fixture predates FEJ sidecars; keep migration coverage
        // explicit rather than fabricating linearized/current values.
        schema_version: MARGDATA_SCHEMA_VERSION_V3,
        aom_sqrt_jacobian: MatrixData::new(0, h_cols, Vec::new()).expect("sqrt shape"),
        aom_sqrt_rhs: Vec::new(),
        aom_abs_h: Some(MatrixData::new(h_rows.len(), h_cols, h_column_major).unwrap()),
        aom_abs_b: Some(array(&input["abs_b"])),
        frame_poses,
        frame_states,
        keyframes: input["kfs_all"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap())
            .collect(),
        kf_to_marg: Vec::new(),
        kfs_all: input["kfs_all"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap())
            .collect(),
        kfs_to_marg: input["kfs_to_marg"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap())
            .collect(),
        aom_order,
        marginalization: MarginalizationTargets::default(),
        prior: None,
        row_counts: [0; 4],
        of_observations: Vec::new(),
        of_images: Vec::new(),
        frame_poses_fej: Default::default(),
        frame_states_fej: Default::default(),
        fej_complete: false,
        measurement_poses: Vec::new(),
        used_imu: input["use_imu"].as_bool().expect("use_imu"),
        provenance_version: "basalt-0f3b2b52-m8f-real-fixture".into(),
    }
}

fn feature_calibration() -> BasaltCalibration {
    let camera =
        DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).expect("camera");
    BasaltCalibration {
        t_imu_cam: vec![
            SE3::identity(),
            SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
        ],
        cameras: vec![camera, camera],
        resolutions: vec![(640, 480), (640, 480)],
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

fn feature_image(frame_id: u64, timestamp_ns: i64, camera_id: u16) -> OfImageData {
    let pixels = (0..128)
        .flat_map(|y| {
            (0..128).map(move |x| {
                let mut value = (x as u32).wrapping_mul(0x9e37_79b9)
                    ^ (y as u32).wrapping_mul(0x85eb_ca6b)
                    ^ ((x as u32) << 16 | y as u32);
                value ^= value >> 13;
                value = value.wrapping_mul(0xc2b2_ae35);
                (value ^ u32::from(camera_id)).wrapping_shr(16) as u16
            })
        })
        .collect();
    OfImageData::new(frame_id, timestamp_ns, camera_id, 128, 128, pixels).expect("image")
}

fn exactly_sixteen_features() -> MapperImageFeatures {
    MapperImageFeatures {
        corners: (0..16)
            .map(|index| Point2::new(320.0 + index as f64, 240.0))
            .collect(),
        corner_angles: vec![0.0; 16],
        descriptors: (0..16).map(|index| [index as u8; 32]).collect(),
        // Identical rays satisfy the known-pose essential gate exactly while
        // the descriptors remain one-to-one, producing exactly 16 inliers.
        rays: vec![[0.0, 0.0, 1.0, 0.0]; 16],
        hashes: vec![0; 16],
        bow_vector: Vec::new(),
    }
}

fn fixture_packet_with_images() -> MargData {
    let mut packet = fixture_packet();
    let first = packet.frame_poses.first().expect("fixture pose");
    let frame_id = first.frame_id;
    let timestamp_ns = first.timestamp_ns;
    packet.of_images = vec![
        feature_image(frame_id, timestamp_ns, 0),
        feature_image(frame_id, timestamp_ns, 1),
        // This timestamp is retained by addMargData but has no installed
        // frame pose, so the pinned detect_keypoints eligibility gate skips it.
        feature_image(frame_id + 1, timestamp_ns + 1, 0),
    ];
    // The synthetic sentinel has no pose/state table entry by design.  Bind
    // its identity through the schema-3 legacy marginalization alias so the
    // strict ingress validator can distinguish it from an orphan image.
    packet.kf_to_marg.push((frame_id + 1, frame_id + 1));
    packet
}

fn fixture_packet_with_two_feature_frames() -> MargData {
    let mut packet = fixture_packet();
    let first = packet.frame_poses.first().expect("fixture pose").clone();
    let second = packet
        .frame_poses
        .get(1)
        .expect("second fixture pose")
        .clone();
    packet.of_images = vec![
        feature_image(first.frame_id, first.timestamp_ns, 0),
        feature_image(first.frame_id, first.timestamp_ns, 1),
        feature_image(second.frame_id, second.timestamp_ns, 0),
        feature_image(second.frame_id, second.timestamp_ns, 1),
        // Retain an ineligible timestamp as a lifecycle sentinel.  Detection
        // must skip it, so it cannot enter the HashBoW key/index sequence.
        feature_image(second.frame_id + 1, second.timestamp_ns + 1, 0),
    ];
    // Keep the ineligible lifecycle sentinel contract-valid without adding a
    // mapper-visible keyframe or state.
    packet
        .kf_to_marg
        .push((second.frame_id + 1, second.frame_id + 1));
    packet
}

#[test]
fn pinned_mh01_packet_is_ingested_and_accumulated() {
    let mut mapper = NfrMapper::default();
    let mut packet = fixture_packet();
    let report = mapper.addMargData(&mut packet).expect("M8f packet");

    assert_eq!(
        report,
        NfrMapperIngestReport {
            input_size: 72,
            output_size: 48,
            accepted: true,
            frame_pose_count: 8,
            relative_pose_factor_count: 7,
            roll_pitch_factor_count: 1,
            image_timestamp_count: 0,
        }
    );
    assert_eq!(mapper.accepted_packets, 1);
    assert_eq!(mapper.frame_poses.len(), 8);
    assert_eq!(mapper.factors.relative_pose.len(), 7);
    assert_eq!(mapper.factors.roll_pitch.len(), 1);
    assert!(mapper.img_data.is_empty());

    // A second packet appends factors while preserving the already installed
    // pose map, matching the persistent upstream mapper vectors/maps.
    let mut second = fixture_packet();
    let second_report = mapper.add_marg_data(&mut second).expect("second packet");
    assert!(second_report.accepted);
    assert_eq!(mapper.accepted_packets, 2);
    assert_eq!(mapper.frame_poses.len(), 8);
    assert_eq!(mapper.factors.relative_pose.len(), 14);
    assert_eq!(mapper.factors.roll_pitch.len(), 2);

    let expected_translation = array(
        &serde_json::from_str::<Value>(FIXTURE).unwrap()["output"]["factors"]["relative_pose"][0]
            ["measurement_translation"],
    );
    assert!(mapper.factors.relative_pose[0]
        .translation
        .iter()
        .zip(expected_translation)
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
}

#[test]
fn pinned_mh01_packet_transitions_into_stateful_detection() {
    let packet = fixture_packet_with_images();
    let first_frame_id = packet.frame_poses[0].frame_id;
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    let mut packet = packet;
    let ingest = mapper.addMargData(&mut packet).expect("M8f packet");
    assert_eq!(
        ingest,
        NfrMapperIngestReport {
            input_size: 72,
            output_size: 48,
            accepted: true,
            frame_pose_count: 8,
            relative_pose_factor_count: 7,
            roll_pitch_factor_count: 1,
            image_timestamp_count: 2,
        }
    );

    let detection = mapper.detectKeypoints().expect("M8c detection");
    assert_eq!(
        detection,
        NfrMapperDetectionReport {
            input_timestamp_count: 2,
            eligible_timestamp_count: 1,
            processed_image_count: 2,
            feature_count: 144,
        }
    );
    assert_eq!(mapper.feature_corners.len(), 2);
    assert!(mapper
        .feature_corners
        .contains_key(&TimeCamId::new(first_frame_id, 0)));
    assert!(mapper
        .feature_corners
        .contains_key(&TimeCamId::new(first_frame_id, 1)));
    assert!(!mapper
        .feature_corners
        .keys()
        .any(|image| image.frame_id == first_frame_id + 1));
    assert!(mapper.feature_corners.values().all(|features| {
        features.corners.len() == features.corner_angles.len()
            && features.corners.len() == features.descriptors.len()
            && features.corners.len() == features.rays.len()
            && features.corners.len() == features.hashes.len()
            && !features.bow_vector.is_empty()
    }));
}

#[test]
fn image_frame_id_is_distinct_from_timestamp_bucket_identity() {
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    let frame_id = 7_u64;
    let timestamp_ns = 1_403_636_579_763_555_584_i64;
    mapper.frame_poses.insert(frame_id, SE3::identity());
    mapper
        .img_data
        .insert(timestamp_ns, vec![feature_image(frame_id, timestamp_ns, 0)]);

    let report = mapper.detect_keypoints().expect("frame/timestamp split");
    assert_eq!(report.input_timestamp_count, 1);
    assert_eq!(report.eligible_timestamp_count, 1);
    assert_eq!(report.processed_image_count, 1);
    assert!(mapper
        .feature_corners
        .contains_key(&TimeCamId::new(frame_id, 0)));
    assert!(!mapper
        .feature_corners
        .contains_key(&TimeCamId::new(timestamp_ns as u64, 0)));
}

fn clean_filter_mapper() -> NfrMapper {
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    for frame_id in 1..=4 {
        mapper.frame_poses.insert(frame_id, SE3::identity());
    }
    let observations = (1..=4)
        .map(|frame_id| MapperObservation {
            image: TimeCamId::new(frame_id, 0),
            feature_id: 0,
            pixel: Point2::new(320.0, 240.0),
        })
        .collect();
    mapper.lmdb.landmarks.insert(
        10,
        MapperLandmark {
            track_id: 10,
            host: TimeCamId::new(1, 0),
            second: TimeCamId::new(2, 0),
            direction: StereographicDirection {
                xy: Point2::new(0.0, 0.0),
            },
            inverse_distance: 1.0,
            observations,
        },
    );
    mapper.lmdb.rebuild_observation_index();
    mapper
}

fn control_mapper_after_full_lifecycle() -> NfrMapper {
    let mut packet = fixture_packet_with_two_feature_frames();
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    mapper.addMargData(&mut packet).expect("M8f control packet");
    let mut feature_config = mapper.feature_config;
    feature_config.min_track_length = 4;
    mapper.set_feature_config(feature_config);
    mapper
        .run_headless(NfrMapperHeadlessConfig {
            num_opt_iter: 3,
            outlier_threshold: 1e9,
            min_num_obs: 4,
            temporal_seed: Some(7),
        })
        .expect("M8f control lifecycle");
    mapper
}

fn rank_deficient_held_out_packet() -> MargData {
    let mut packet = fixture_packet();
    let h = packet.aom_abs_h.as_mut().expect("absolute system");
    h.data.fill(0.0);
    packet
}

/// Stable FNV-1a snapshot over the public mapper state used by the production
/// output boundary.  Maps are BTreeMap/BTreeSet-backed, and the JSON object
/// field order is explicit, so this hash is independent of allocator layout.
fn canonical_mapper_snapshot_hash(mapper: &NfrMapper) -> u64 {
    let frame_timestamps = mapper
        .frame_timestamps
        .iter()
        .map(|(&frame_id, &timestamp_ns)| json!([frame_id, timestamp_ns]))
        .collect::<Vec<_>>();
    let feature_matches = mapper
        .feature_matches
        .iter()
        .map(|((left, right), match_data)| {
            json!({
                "left": [left.frame_id, left.cam_id],
                "right": [right.frame_id, right.cam_id],
                "inliers": match_data.inliers,
            })
        })
        .collect::<Vec<_>>();
    let feature_tracks = mapper
        .feature_tracks
        .iter()
        .map(|(&track_id, observations)| {
            json!({
                "track_id": track_id,
                "observations": observations
                    .iter()
                    .map(|(image, &feature_id)| {
                        json!([image.frame_id, image.cam_id, feature_id])
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let lmdb_observations = mapper
        .lmdb
        .observations
        .iter()
        .map(|(host, targets)| {
            json!({
                "host": [host.frame_id, host.cam_id],
                "targets": targets
                    .iter()
                    .map(|(target, track_ids)| {
                        json!({
                            "target": [target.frame_id, target.cam_id],
                            "track_ids": track_ids.iter().copied().collect::<Vec<_>>(),
                        })
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let snapshot = json!({
        "accepted_packets": mapper.accepted_packets,
        "frame_poses": mapper.frame_pose_records(),
        "frame_timestamps": frame_timestamps,
        "factors": mapper.factors,
        "feature_matches": feature_matches,
        "feature_tracks": feature_tracks,
        "lmdb_observations": lmdb_observations,
        "result": mapper.result(),
    });
    // Floats to six significant digits: the hash pins the mapper's order
    // and values, not the platform libm's last bits (Ubuntu 24.04's and
    // 26.04's glibc differ from the tenth digit on).
    let rounded = round_floats(&snapshot, 6);
    let bytes = serde_json::to_vec(&rounded).expect("canonical mapper snapshot");
    bytes
        .into_iter()
        .fold(1469598103934665603_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1099511628211_u64)
        })
}

/// `value` with every float rounded to `digits` significant digits.
fn round_floats(value: &Value, digits: usize) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            let float = number.as_f64().expect("an f64 number");
            let rounded: f64 = format!("{:.*e}", digits - 1, float)
                .parse()
                .expect("a formatted float");
            // One zero, whatever its sign.
            json!(if rounded == 0.0 { 0.0 } else { rounded })
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| round_floats(v, digits)).collect())
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), round_floats(v, digits)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[test]
fn held_out_control_lifecycle_binds_order_and_snapshot_hash() {
    let first = control_mapper_after_full_lifecycle();
    let second = control_mapper_after_full_lifecycle();

    assert_eq!(first.result(), second.result());
    assert_eq!(first.trajectory_euroc(), second.trajectory_euroc());
    assert_eq!(first.trajectory_tum(), second.trajectory_tum());
    assert_eq!(first.feature_tracks, second.feature_tracks);
    assert_eq!(first.lmdb, second.lmdb);
    assert_eq!(
        first.frame_pose_records(),
        second.frame_pose_records(),
        "control pose order/output must be deterministic"
    );
    assert_eq!(
        first.factors, second.factors,
        "control factor order/output must be deterministic"
    );
    assert_eq!(
        canonical_mapper_snapshot_hash(&first),
        canonical_mapper_snapshot_hash(&second),
        "control canonical snapshot hash must be stable"
    );
    assert_eq!(
        canonical_mapper_snapshot_hash(&first),
        // Re-pinned for the LM retry fix (retries re-solve with damped
        // landmark blocks); this control fixture exercises rejected trials.
        // Re-pinned again when the floats were rounded (Ubuntu 26.04).
        13_691_810_757_751_788_245,
        "held-out control canonical snapshot changed"
    );
    assert_eq!(first.result().poses.len(), 8);
    assert_eq!(first.result().landmarks.len(), 72);
    assert_eq!(
        first.feature_tracks.keys().copied().collect::<Vec<_>>(),
        (0..72).collect::<Vec<_>>()
    );
    assert_eq!(first.lmdb.num_observations(), 288);
}

#[test]
fn held_out_rank_deficient_packet_rejects_without_state_or_landmark_mutation() {
    let mut mapper = control_mapper_after_full_lifecycle();
    let before_hash = canonical_mapper_snapshot_hash(&mapper);
    let before_result = mapper.result();
    let before_euroc = mapper.trajectory_euroc();
    let before_tum = mapper.trajectory_tum();
    let before_mapper = mapper.clone();

    let mut held_out = rank_deficient_held_out_packet();
    let report = mapper
        .addMargData(&mut held_out)
        .expect("rank-deficient packet is a reported rejection");

    assert!(!report.accepted);
    assert_eq!(report.frame_pose_count, before_mapper.frame_poses.len());
    assert_eq!(
        report.relative_pose_factor_count,
        before_mapper.factors.relative_pose.len()
    );
    assert_eq!(
        report.roll_pitch_factor_count,
        before_mapper.factors.roll_pitch.len()
    );
    assert_eq!(canonical_mapper_snapshot_hash(&mapper), before_hash);
    assert_eq!(mapper.result(), before_result);
    assert_eq!(mapper.trajectory_euroc(), before_euroc);
    assert_eq!(mapper.trajectory_tum(), before_tum);
    assert_eq!(mapper.frame_poses, before_mapper.frame_poses);
    assert_eq!(mapper.frame_timestamps, before_mapper.frame_timestamps);
    assert_eq!(mapper.factors, before_mapper.factors);
    assert_eq!(mapper.feature_tracks, before_mapper.feature_tracks);
    assert_eq!(mapper.lmdb, before_mapper.lmdb);
    assert_eq!(mapper.accepted_packets, before_mapper.accepted_packets);
}

#[test]
fn filter_outliers_preserves_source_delete_order_and_min_obs_gate() {
    let mut remove_landmark = clean_filter_mapper();
    remove_landmark
        .lmdb
        .landmarks
        .get_mut(&10)
        .unwrap()
        .observations[1]
        .pixel
        .x += 100.0;
    let report = remove_landmark
        .filterOutliers(3.0, 4)
        .expect("filter outliers");
    assert_eq!(report.candidate_landmark_count, 1);
    assert_eq!(report.outlier_observation_count, 1);
    assert_eq!(report.removed_landmark_count, 1);
    assert_eq!(report.removed_observation_count, 0);
    assert_eq!(remove_landmark.lmdb.num_landmarks(), 0);

    let mut remove_observation = clean_filter_mapper();
    remove_observation
        .lmdb
        .landmarks
        .get_mut(&10)
        .unwrap()
        .observations[1]
        .pixel
        .x += 100.0;
    let report = remove_observation
        .filter_outliers(3.0, 3)
        .expect("filter observation");
    assert_eq!(report.removed_landmark_count, 0);
    assert_eq!(report.removed_observation_count, 1);
    assert_eq!(remove_observation.lmdb.num_landmarks(), 1);
    assert_eq!(remove_observation.lmdb.num_observations(), 3);
}

#[test]
fn headless_mapper_preserves_source_stage_order_and_continuity() {
    let mut packet = fixture_packet_with_two_feature_frames();
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    mapper.addMargData(&mut packet).expect("M8f packet");
    let mut feature_config = mapper.feature_config;
    feature_config.min_track_length = 4;
    mapper.set_feature_config(feature_config);

    let report = mapper
        .run_headless(NfrMapperHeadlessConfig {
            num_opt_iter: 3,
            outlier_threshold: 1e9,
            min_num_obs: 4,
            temporal_seed: Some(7),
        })
        .expect("headless M8 lifecycle");
    assert_eq!(report.detection.processed_image_count, 4);
    assert_eq!(report.stereo.accepted_pair_count, 2);
    assert_eq!(report.match_all.accepted_pair_count, 4);
    assert_eq!(report.tracks.exported_track_count, 72);
    assert_eq!(report.setup.accepted_track_count, 72);
    assert_eq!(report.first_optimize.requested_iterations, 3);
    assert_eq!(report.second_optimize.requested_iterations, 3);
    assert_eq!(report.filter.removed_landmark_count, 0);
    assert_eq!(
        report.initial_points.points.len(),
        report.setup.observation_count
    );
    assert_eq!(
        report.filtered_points.points.len(),
        report.filter.after_observation_count
    );
    assert_eq!(
        report.final_points.points.len(),
        report.filter.after_observation_count
    );
    assert_eq!(report.result.poses.len(), mapper.frame_poses.len());
    assert_eq!(
        report
            .stages
            .iter()
            .map(|stage| stage.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "setup_opt_get_points",
            "optimize",
            "filter_get_points",
            "optimize_filtered",
            "final_get_points_result",
        ]
    );
}

#[test]
fn pinned_mh01_packet_matches_stereo_with_source_gates() {
    let mut packet = fixture_packet_with_images();
    let first_frame_id = packet.frame_poses[0].frame_id;
    let second_frame = packet.frame_poses[1].clone();
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    mapper.addMargData(&mut packet).expect("M8f packet");
    mapper.detect_keypoints().expect("M8c detection");

    // Install a second real-packet timestamp with exactly 16 one-to-one
    // inliers.  The source's strict `inliers.size() > 16` gate must reject
    // this pair while preserving the first accepted pair.
    mapper.img_data.insert(
        second_frame.timestamp_ns,
        vec![
            feature_image(second_frame.frame_id, second_frame.timestamp_ns, 0),
            feature_image(second_frame.frame_id, second_frame.timestamp_ns, 1),
        ],
    );
    mapper.feature_corners.insert(
        TimeCamId::new(second_frame.frame_id, 0),
        exactly_sixteen_features(),
    );
    mapper.feature_corners.insert(
        TimeCamId::new(second_frame.frame_id, 1),
        exactly_sixteen_features(),
    );

    let report = mapper.matchStereo().expect("M8c stereo matching");
    assert_eq!(
        report,
        NfrMapperStereoReport {
            input_timestamp_count: 3,
            attempted_timestamp_count: 3,
            missing_feature_timestamp_count: 1,
            raw_match_count: 88,
            essential_inlier_count: 88,
            accepted_pair_count: 1,
            total_pair_count: 1,
        }
    );
    assert_eq!(
        mapper
            .feature_matches
            .get(&(
                TimeCamId::new(first_frame_id, 0),
                TimeCamId::new(first_frame_id, 1),
            ))
            .expect("accepted stereo pair")
            .inliers
            .len(),
        72
    );
    let full_match = mapper
        .feature_match_data
        .get(&(
            TimeCamId::new(first_frame_id, 0),
            TimeCamId::new(first_frame_id, 1),
        ))
        .expect("full accepted stereo payload");
    assert_eq!(full_match.matches.len(), 72);
    assert_eq!(full_match.inliers.len(), 72);
    assert!((full_match.t_i_j.translation.x - 0.1).abs() < 1e-12);
    assert!(!mapper
        .feature_matches
        .keys()
        .any(|(left, _)| left.frame_id != first_frame_id));
    assert!(!mapper
        .feature_matches
        .keys()
        .any(|(left, _)| { left.frame_id == second_frame.frame_id }));
    assert_eq!(mapper.feature_corners.len(), 6);
}

#[test]
fn pinned_mh01_packet_match_all_preserves_bow_order_and_payload() {
    let mut packet = fixture_packet_with_two_feature_frames();
    let first_frame_id = packet.frame_poses[0].frame_id;
    let second_frame_id = packet.frame_poses[1].frame_id;
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    mapper.addMargData(&mut packet).expect("M8f packet");
    mapper.detect_keypoints().expect("M8c detection");
    assert_eq!(mapper.feature_corners.len(), 4);
    // `match_all` follows the pinned hardcoded descriptor controls (70, 1.2)
    // rather than the stereo-configurable values.  Change the public config
    // after detection so this source distinction is exercised by the fixture.
    let mut altered_config = mapper.feature_config;
    altered_config.max_hamming = 0;
    altered_config.second_best_ratio = 0.01;
    mapper.set_feature_config(altered_config);

    let report = mapper.match_all_seeded(7);
    assert_eq!(report.input_feature_count, 4);
    assert_eq!(report.query_count, 4);
    assert_eq!(report.candidate_pair_count, report.candidate_pairs.len());
    assert_eq!(report.candidate_pair_count, 4);
    assert_eq!(report.raw_gate_rejected_pair_count, 0);
    assert_eq!(report.ransac_attempt_count, report.candidate_pair_count);
    assert_eq!(report.geometric_rejected_pair_count, 0);
    assert_eq!(report.accepted_pair_count, 4);
    assert_eq!(report.total_pair_count, 4);
    assert!(report
        .candidate_pairs
        .iter()
        .all(|((left, right), score)| left.frame_id == second_frame_id
            && right.frame_id == first_frame_id
            && *score > 0.04));
    assert_eq!(
        report
            .candidate_pairs
            .iter()
            .map(|((left, right), _)| (left.cam_id, right.cam_id))
            .collect::<Vec<_>>(),
        vec![(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    assert!(report.raw_match_count > 4 * mapper.feature_config.min_matches);

    for ((left, right), _) in &report.candidate_pairs {
        let projection = mapper
            .feature_matches
            .get(&(*left, *right))
            .expect("accepted track projection");
        let full = mapper
            .feature_match_data
            .get(&(*left, *right))
            .expect("accepted full source payload");
        assert_eq!(projection.inliers, full.inliers);
        assert_eq!(full.matches.len(), 72);
        assert_eq!(full.inliers.len(), 72);
        assert!((full.t_i_j.translation.norm() - 1.0).abs() < 1e-9);
    }
    assert!(mapper
        .feature_matches
        .keys()
        .all(|(left, right)| left.frame_id == second_frame_id && right.frame_id == first_frame_id));
    assert!(mapper
        .feature_matches
        .keys()
        .all(|(left, right)| left.frame_id != right.frame_id));
    assert!(second_frame_id > first_frame_id);

    // The source query passes the configured num-frames value directly to
    // HashBow.  A one-result query retains only the first (ID-ordered) older
    // camera for each of the two second-frame images.
    let mut limited = mapper.clone();
    limited.feature_matches.clear();
    limited.feature_match_data.clear();
    limited.feature_config.match_window = 1;
    let limited_report = limited.match_all_seeded(7);
    assert_eq!(limited_report.candidate_pair_count, 2);
    assert_eq!(limited_report.ransac_attempt_count, 2);
    assert_eq!(limited_report.accepted_pair_count, 2);

    // Equality at the threshold must reject a candidate: the source gate is
    // `score > mapper_frames_to_match_threshold`, never `>=`.
    let mut thresholded = mapper.clone();
    thresholded.feature_matches.clear();
    thresholded.feature_match_data.clear();
    thresholded.feature_config.frames_to_match_threshold = report.candidate_pairs[0].1;
    let thresholded_report = thresholded.match_all_seeded(7);
    assert_eq!(thresholded_report.candidate_pair_count, 0);
    assert_eq!(thresholded_report.accepted_pair_count, 0);
    assert_eq!(thresholded_report.total_pair_count, 0);
}

#[test]
fn pinned_mh01_packet_build_tracks_keeps_roots_and_rejects_conflicts() {
    let mut packet = fixture_packet_with_two_feature_frames();
    let first_frame_id = packet.frame_poses[0].frame_id;
    let second_frame_id = packet.frame_poses[1].frame_id;
    let mut mapper = NfrMapper::with_calibration(Default::default(), feature_calibration());
    mapper.addMargData(&mut packet).expect("M8f packet");
    mapper.detect_keypoints().expect("M8c detection");

    let stereo = mapper.matchStereo().expect("M8c stereo matching");
    assert_eq!(
        stereo,
        NfrMapperStereoReport {
            input_timestamp_count: 3,
            attempted_timestamp_count: 3,
            missing_feature_timestamp_count: 1,
            raw_match_count: 144,
            essential_inlier_count: 144,
            accepted_pair_count: 2,
            total_pair_count: 2,
        }
    );

    // match_all's descriptor controls remain source-hardcoded, while the
    // track filter is configured to accept this four-camera fixture's tracks.
    let mut config = mapper.feature_config;
    config.max_hamming = 0;
    config.second_best_ratio = 0.01;
    config.min_track_length = 4;
    mapper.set_feature_config(config);
    let temporal = mapper.matchAllSeeded(7);
    assert_eq!(temporal.candidate_pair_count, 4);
    assert_eq!(temporal.accepted_pair_count, 4);
    assert_eq!(temporal.total_pair_count, 6);

    let report = mapper.buildTracks();
    assert_eq!(report.input_pair_count, 6);
    assert_eq!(report.inlier_match_count, 432);
    assert_eq!(report.node_count, 288);
    assert_eq!(report.component_count_before, 72);
    assert_eq!(report.component_count_after, 72);
    assert!(report.rejected_conflict_ids.is_empty());
    assert!(report.rejected_short_ids.is_empty());
    assert!(report.rejected_track_ids.is_empty());
    assert_eq!(report.total_track_obs_count, 288);
    assert_eq!(report.average_track_length, Some(4.0));
    assert_eq!(report.exported_track_count, 72);
    assert_eq!(report.filter.track_length_histogram.get(&4), Some(&72));
    assert_eq!(mapper.feature_tracks.len(), 72);
    assert_eq!(
        mapper.feature_tracks.keys().copied().collect::<Vec<_>>(),
        (0..72).collect::<Vec<_>>()
    );
    for observations in mapper.feature_tracks.values() {
        assert_eq!(observations.len(), 4);
        assert!(observations.contains_key(&TimeCamId::new(first_frame_id, 0)));
        assert!(observations.contains_key(&TimeCamId::new(first_frame_id, 1)));
        assert!(observations.contains_key(&TimeCamId::new(second_frame_id, 0)));
        assert!(observations.contains_key(&TimeCamId::new(second_frame_id, 1)));
    }
    // A second pass rebuilds and replaces the persistent export in the same
    // order, rather than accumulating duplicate tracks.
    assert_eq!(mapper.build_tracks(), report);

    let setup = mapper.setupOpt().expect("M8d setup_opt");
    assert_eq!(setup.input_track_count, 72);
    assert_eq!(setup.attempted_track_count, 72);
    assert_eq!(setup.accepted_track_count, 72);
    assert_eq!(setup.skipped_short_track_count, 0);
    assert_eq!(setup.observation_count, 288);
    assert_eq!(setup.candidate_attempt_count, 120);
    assert_eq!(
        setup
            .rejection_counts
            .get(&SetupOptRejectReason::InverseDistanceNonPositive),
        Some(&48)
    );
    assert_eq!(mapper.lmdb.num_landmarks(), 72);
    assert_eq!(mapper.lmdb.num_observations(), 288);
    assert_eq!(mapper.lmdb.num_hosts(), 1);
    let host = TimeCamId::new(first_frame_id, 0);
    let host_targets = mapper.lmdb.observations.get(&host).expect("host index");
    assert_eq!(host_targets.len(), 4);
    for target in [
        TimeCamId::new(first_frame_id, 0),
        TimeCamId::new(first_frame_id, 1),
        TimeCamId::new(second_frame_id, 0),
        TimeCamId::new(second_frame_id, 1),
    ] {
        assert_eq!(host_targets.get(&target).map(BTreeSet::len), Some(72));
    }
    assert!(mapper.lmdb.landmarks.values().all(|landmark| {
        landmark.host == host
            && landmark.observations.len() == 4
            && landmark.inverse_distance.is_finite()
            && landmark.inverse_distance > 0.0
            && landmark.inverse_distance <= 2.0
    }));

    // Continue the same real packet through the stateful M8e handoff.  The
    // source fixture enables LM and uses three iterations here so both the
    // accepted-step persistence and the exact lambda/lambda-vee updates are
    // exercised without relying on a wall-clock termination condition.
    mapper.set_optimize_config(GlobalBaConfig {
        use_lm: true,
        max_iterations: 3,
        ..GlobalBaConfig::default()
    });
    let before_pose_count = mapper.frame_poses.len();
    let before_landmark_count = mapper.lmdb.num_landmarks();
    let optimize = mapper.optimize(3).expect("M8e optimize");
    assert_eq!(optimize.requested_iterations, 3);
    assert_eq!(optimize.pose_count, before_pose_count);
    assert_eq!(optimize.landmark_count, before_landmark_count);
    assert!(optimize.initial_cost.is_finite());
    assert!(optimize.final_cost.is_finite());
    assert!(optimize.final_cost <= optimize.initial_cost);
    assert!(optimize.accepted_step_count > 0);
    assert_eq!(optimize.accepted_step_count, 2);
    assert_eq!(optimize.rejected_trial_count, 10);
    assert_eq!(optimize.trace.len(), 3);
    assert_eq!(
        optimize
            .trace
            .iter()
            .map(|iteration| iteration.trials.len())
            .collect::<Vec<_>>(),
        vec![1, 1, 10]
    );
    assert!(optimize.trace[0].trials[0].accepted);
    assert!(optimize.trace[1].trials[0].accepted);
    assert!(optimize.trace[2].trials.iter().all(|trial| !trial.accepted));
    assert_eq!(optimize.initial_lambda, 1e-32);
    assert_eq!(optimize.initial_lambda_vee, 2.0);
    assert_eq!(optimize.final_lambda, 3.602879701896397e-16);
    assert_eq!(optimize.final_lambda_vee, 2048.0);
    assert_eq!(mapper.lmdb.num_landmarks(), before_landmark_count);
    assert_eq!(mapper.lmdb.num_observations(), 288);
    assert!(mapper.frame_poses.values().all(|pose| {
        pose.translation.iter().all(|value| value.is_finite())
            && pose.rotation.coords.iter().all(|value| value.is_finite())
    }));
    // Post-BA lifecycle boundary: the source getter exposes the same
    // optimized pose map, diagnostics evaluate the recovered factor rows, and
    // result/trajectory snapshots retain frame/timestamp and pose ordering.
    assert_eq!(mapper.get_frame_poses().len(), before_pose_count);
    assert_eq!(
        mapper.get_frame_poses().keys().next(),
        Some(&first_frame_id)
    );
    assert!(mapper.computeRelPose().is_finite());
    assert!(mapper.computeRollPitch().is_finite());
    let result = mapper.result();
    assert_eq!(result.poses.len(), before_pose_count);
    assert_eq!(result.landmarks.len(), before_landmark_count);
    assert_eq!(result.poses[0].frame_id, first_frame_id);
    assert_eq!(result.poses[0].timestamp_ns, first_frame_id as i64);
    assert_eq!(
        result.landmarks[0].track_id,
        mapper.lmdb.landmarks.keys().next().copied().unwrap()
    );
    assert_eq!(mapper.trajectory_poses().len(), before_pose_count);
    let trajectory_euroc = mapper.trajectory_euroc();
    assert_eq!(trajectory_euroc.lines().count(), before_pose_count + 1);
    assert!(trajectory_euroc.starts_with(
        "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n"
    ));
    let trajectory_tum = mapper.trajectory_tum();
    assert_eq!(trajectory_tum.lines().count(), before_pose_count + 1);
    assert!(trajectory_tum.starts_with("# timestamp tx ty tz qx qy qz qw\n"));

    // Add one cross-feature edge to an existing pair.  It merges two valid
    // components, creates a duplicate image in that component, and must
    // reject the complete component while leaving the other 70 roots
    // unrenumbered.
    let mut conflicted = mapper.clone();
    conflicted.feature_matches.insert(
        (
            TimeCamId::new(first_frame_id, 0),
            TimeCamId::new(second_frame_id, 0),
        ),
        MatchData::new(vec![(0, 1)]),
    );
    let conflict_report = conflicted.build_tracks();
    assert_eq!(conflict_report.input_pair_count, 7);
    assert_eq!(conflict_report.inlier_match_count, 433);
    assert_eq!(conflict_report.node_count, 288);
    assert_eq!(conflict_report.component_count_before, 71);
    assert_eq!(conflict_report.component_count_after, 70);
    assert_eq!(conflict_report.rejected_conflict_ids, vec![0]);
    // The pinned Filter traversal stops recording a component after its
    // first conflict, leaving the root's retained image set length at one;
    // that same root therefore appears in both rejection buckets.
    assert_eq!(conflict_report.rejected_short_ids, vec![0]);
    assert_eq!(conflict_report.rejected_track_ids, vec![0]);
    assert_eq!(conflict_report.total_track_obs_count, 280);
    assert_eq!(conflict_report.average_track_length, Some(4.0));
    assert_eq!(conflict_report.exported_track_count, 70);
    assert!(!conflicted.feature_tracks.contains_key(&0));
    assert!(!conflicted.feature_tracks.contains_key(&1));
    assert!(conflicted.feature_tracks.keys().copied().eq(2..72));

    // Empty input is a valid source state; expose a safe `None` diagnostic
    // instead of dividing by zero while exporting no tracks.
    let mut empty = mapper.clone();
    empty.feature_matches.clear();
    let empty_report = empty.build_tracks();
    assert_eq!(empty_report.node_count, 0);
    assert_eq!(empty_report.exported_track_count, 0);
    assert_eq!(empty_report.total_track_obs_count, 0);
    assert_eq!(empty_report.average_track_length, None);
    assert!(empty.feature_tracks.is_empty());
    let empty_setup = empty.setup_opt().expect("empty M8d setup_opt");
    assert_eq!(empty_setup.input_track_count, 0);
    assert_eq!(empty.lmdb.num_landmarks(), 0);
    assert_eq!(empty.lmdb.num_observations(), 0);
}
