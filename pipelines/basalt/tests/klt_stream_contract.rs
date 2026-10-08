use std::collections::BTreeSet;

use nalgebra::Vector3;
use serde::Deserialize;
use visloc_basalt::{
    BasaltCalibration, DirectKltConfig, DirectKltStream, GridFastConfig, KltFailure, RawU16Image,
    RejectReason, StereoFrame, TrackStage,
};
use visloc_core::geometry::SE3;

/// Small deterministic scene used by the direct-stream contract tests.  The
/// high-contrast square lattice gives FAST corners while the intra-cell ramp
/// makes the 52-sample KLT patches observable at every pyramid level.
const WIDTH: usize = 96;
const HEIGHT: usize = 96;
const BLOCK: usize = 8;

#[derive(Debug, Deserialize)]
struct SyntheticFixture {
    width: usize,
    height: usize,
    block: usize,
    frames: Vec<SyntheticFrame>,
    failure_frame: FailureFrame,
}

#[derive(Debug, Deserialize)]
struct SyntheticFrame {
    frame_id: u64,
    timestamp_ns: i64,
    cam0_dx: isize,
    cam0_dy: isize,
    cam1_dx: isize,
    cam1_dy: isize,
}

#[derive(Debug, Deserialize)]
struct FailureFrame {
    frame_id: u64,
    timestamp_ns: i64,
}

fn fixture() -> SyntheticFixture {
    let fixture: SyntheticFixture =
        serde_json::from_str(include_str!("fixtures/klt_stream_synthetic.json")).unwrap();
    assert_eq!(
        (fixture.width, fixture.height, fixture.block),
        (WIDTH, HEIGHT, BLOCK)
    );
    assert_eq!(fixture.frames.len(), 3);
    fixture
}

fn scene_pixel(x: usize, y: usize) -> u16 {
    let checker = ((x / BLOCK) + (y / BLOCK)) % 2;
    let base = if checker == 0 { 8_000_u32 } else { 52_000_u32 };
    let ramp = ((x * 37 + y * 53 + (x * y) % 17) % 31) as u32 * 120;
    (base + ramp).min(u16::MAX as u32) as u16
}

fn shifted_scene(dx: isize, dy: isize) -> RawU16Image {
    RawU16Image::from_fn(WIDTH, HEIGHT, |x, y| {
        let source_x = x as isize - dx;
        let source_y = y as isize - dy;
        if source_x < 0 || source_y < 0 || source_x >= WIDTH as isize || source_y >= HEIGHT as isize
        {
            0
        } else {
            scene_pixel(source_x as usize, source_y as usize)
        }
    })
    .unwrap()
}

fn fixture_frame(index: usize) -> StereoFrame {
    let fixture = fixture();
    let frame = &fixture.frames[index];
    StereoFrame::new(
        frame.frame_id,
        frame.timestamp_ns,
        shifted_scene(frame.cam0_dx, frame.cam0_dy),
        Some(shifted_scene(frame.cam1_dx, frame.cam1_dy)),
    )
}

fn black_failure_frame() -> StereoFrame {
    let fixture = fixture();
    StereoFrame::new(
        fixture.failure_frame.frame_id,
        fixture.failure_frame.timestamp_ns,
        RawU16Image::new(WIDTH, HEIGHT, vec![0; WIDTH * HEIGHT]).unwrap(),
        Some(RawU16Image::new(WIDTH, HEIGHT, vec![0; WIDTH * HEIGHT]).unwrap()),
    )
}

fn calibration() -> BasaltCalibration {
    let camera = visloc_basalt::DoubleSphereCamera::new(
        42.0,
        42.0,
        48.0,
        48.0,
        0.0,
        0.5,
        WIDTH as u32,
        HEIGHT as u32,
    )
    .unwrap();
    BasaltCalibration {
        t_imu_cam: vec![
            SE3::identity(),
            SE3::new(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(0.1, 0.0, 0.0),
            ),
        ],
        cameras: vec![camera, camera],
        resolutions: vec![(WIDTH as u32, HEIGHT as u32); 2],
        calib_accel_bias: vec![0.0; 9],
        calib_gyro_bias: vec![0.0; 12],
        imu_update_rate_hz: 200.0,
        accel_noise_std: Vector3::repeat(0.01),
        gyro_noise_std: Vector3::repeat(0.01),
        accel_bias_std: Vector3::repeat(0.01),
        gyro_bias_std: Vector3::repeat(0.01),
        t_mocap_world: SE3::identity(),
        t_imu_marker: SE3::identity(),
        mocap_time_offset_ns: 0,
        mocap_to_imu_offset_ns: 0,
        cam_time_offset_ns: 0,
    }
}

const fn config() -> DirectKltConfig {
    DirectKltConfig {
        imu_seed_rotation: false,
        pyramid_levels: 2,
        max_iterations: 5,
        fb_squared_threshold: 0.04,
        essential_residual_threshold: 0.005,
        fast: GridFastConfig {
            cell_size: 24,
            points_per_cell: 1,
            threshold: 40,
            min_threshold: 5,
            edge_threshold: 5,
        },
    }
}

fn stage_positions(trace: &[TrackStage], stage: TrackStage) -> Vec<usize> {
    trace
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| (*candidate == stage).then_some(index))
        .collect()
}

#[test]
fn direct_stream_preserves_upstream_stage_order_and_stereo_emit() {
    let mut stream = DirectKltStream::new(calibration(), config()).unwrap();
    let output = stream.process_frame(fixture_frame(0)).unwrap();

    assert!(!output.created_track_ids.is_empty());
    assert_eq!(output.retained_track_ids, Vec::<u64>::new());
    let cam0_ids: BTreeSet<_> = output
        .observations
        .iter()
        .filter(|observation| observation.camera_id == 0)
        .map(|observation| observation.track_id)
        .collect();
    assert_eq!(cam0_ids.len(), output.created_track_ids.len());
    assert!(output
        .observations
        .iter()
        .any(|observation| observation.camera_id == 1));
    assert!(output.observations.iter().all(|observation| {
        observation.camera_id == 0 || cam0_ids.contains(&observation.track_id)
    }));

    let required_order = [
        TrackStage::FrameForwardSe2Ic,
        TrackStage::FrameBackwardSe2Ic,
        TrackStage::FrameFbSquared,
        TrackStage::Cam0GridFastReplenish,
        TrackStage::StereoForwardSe2Ic,
        TrackStage::StereoBackwardSe2Ic,
        TrackStage::StereoFbSquared,
        TrackStage::DsBearingEssentialResidual,
        TrackStage::Emit,
    ];
    let mut last = None;
    for stage in required_order {
        let position = stage_positions(&output.stage_trace, stage)
            .into_iter()
            .next()
            .expect("required direct-stream stage missing");
        assert!(last.is_none_or(|previous| previous < position));
        last = Some(position);
    }
}

#[test]
fn direct_stream_keeps_ids_across_three_frames_and_replenishes_monotonically() {
    let mut stream = DirectKltStream::new(calibration(), config()).unwrap();
    let first = stream.process_frame(fixture_frame(0)).unwrap();
    let first_ids: BTreeSet<_> = first.created_track_ids.iter().copied().collect();
    assert!(!first_ids.is_empty());

    let second = stream.process_frame(fixture_frame(1)).unwrap();
    let second_ids: BTreeSet<_> = second
        .retained_track_ids
        .iter()
        .chain(second.created_track_ids.iter())
        .copied()
        .collect();
    assert!(second
        .retained_track_ids
        .iter()
        .all(|track_id| first_ids.contains(track_id)));
    assert!(second
        .created_track_ids
        .iter()
        .all(|track_id| !first_ids.contains(track_id)));
    assert!(second_ids
        .iter()
        .all(|track_id| { *track_id < stream.next_track_id() }));

    let third = stream.process_frame(fixture_frame(2)).unwrap();
    assert!(third
        .retained_track_ids
        .iter()
        .all(|track_id| second_ids.contains(track_id)));
    assert!(
        third
            .created_track_ids
            .iter()
            .all(|track_id| *track_id
                >= stream.next_track_id() - third.created_track_ids.len() as u64)
    );
    assert!(third
        .observations
        .iter()
        .all(|observation| observation.frame_id == 2));
}

#[test]
fn direct_stream_classifies_forward_loss_and_accumulates_rejects() {
    let mut stream = DirectKltStream::new(calibration(), config()).unwrap();
    let first = stream.process_frame(fixture_frame(0)).unwrap();
    assert!(!first.created_track_ids.is_empty());

    // An all-black target has no usable normalized patch.  The old feature
    // IDs are rejected in the forward KLT stage, while the stream remains
    // usable and records the classification rather than silently relabeling.
    let failed = stream.process_frame(black_failure_frame()).unwrap();
    assert_eq!(
        failed.rejected_track_ids.len(),
        first.created_track_ids.len()
    );
    assert!(failed
        .reject_counters
        .iter()
        .any(|(reason, count)| matches!(
            reason,
            RejectReason::FrameForward(
                KltFailure::TargetNoValidSamples | KltFailure::SourcePatchInvalid
            )
        ) && *count > 0));
    assert!(stream.cumulative_reject_counters().total() >= failed.reject_counters.total());
}

#[test]
fn direct_stream_records_stereo_essential_rejection_separately() {
    let mut calibration = calibration();
    calibration.t_imu_cam[1].translation.x = 0.1;
    let mut stream = DirectKltStream::new(calibration, config()).unwrap();
    let output = stream
        .process_frame(StereoFrame::new(
            fixture().frames[0].frame_id,
            fixture().frames[0].timestamp_ns,
            shifted_scene(0, 0),
            Some(shifted_scene(6, 0)),
        ))
        .unwrap();
    assert!(output
        .reject_counters
        .iter()
        .any(|(reason, count)| *reason == RejectReason::StereoEssentialResidual && *count > 0));
    assert!(
        output
            .observations
            .iter()
            .filter(|observation| observation.camera_id == 1)
            .count()
            < output.created_track_ids.len()
    );
}

#[test]
fn direct_stream_runs_monocular_without_cam1_or_stereo_stages() {
    // A cam0-only calibration plus cam1-less frames is the `--mono` frontend:
    // temporal KLT and replenishment run as usual, no stereo stage executes,
    // and only camera-0 observations are emitted.
    let mono_calibration = calibration().retain_cameras(1).unwrap();
    let mut stream = DirectKltStream::new(mono_calibration, config()).unwrap();
    let fixture = fixture();
    let mut previous_ids = BTreeSet::new();
    for (index, frame) in fixture.frames.iter().enumerate() {
        let output = stream
            .process_frame(StereoFrame::new(
                frame.frame_id,
                frame.timestamp_ns,
                shifted_scene(frame.cam0_dx, frame.cam0_dy),
                None,
            ))
            .unwrap();
        assert!(!output.observations.is_empty());
        assert!(output
            .observations
            .iter()
            .all(|observation| observation.camera_id == 0));
        // `stage_trace` is the static upstream stage order, so check the
        // reject counters instead: no stereo stage may ever classify a track.
        assert!(output.reject_counters.iter().all(|(reason, count)| {
            *count == 0
                || !matches!(
                    reason,
                    RejectReason::ExistingStereoForward(..)
                        | RejectReason::ExistingStereoBackward(..)
                        | RejectReason::ExistingStereoFbSquared
                        | RejectReason::StereoForward(..)
                        | RejectReason::StereoBackward(..)
                        | RejectReason::StereoFbSquared
                        | RejectReason::StereoBearingInvalid
                        | RejectReason::StereoEssentialResidual
                )
        }));
        if index > 0 {
            assert!(!output.retained_track_ids.is_empty());
            assert!(output
                .retained_track_ids
                .iter()
                .all(|track_id| previous_ids.contains(track_id)));
        }
        previous_ids = output
            .observations
            .iter()
            .map(|observation| observation.track_id)
            .collect();
    }

    // A cam1 image without a camera-1 calibration is rejected rather than
    // silently ignored.
    let mut stream =
        DirectKltStream::new(calibration().retain_cameras(1).unwrap(), config()).unwrap();
    assert!(stream.process_frame(fixture_frame(0)).is_err());
}
