//! Online EuRoC VI-SLAM demo: VIO thread + concurrent mapper thread.
//!
//! Single process, one run over the sensor stream. The VIO thread (this
//! process's main thread) consumes images/IMU at either dataset rate
//! (`--realtime`) or as fast as possible (default), and for every
//! `MargData` mapper packet moves it (no JSON/base64 round trip --
//! `EstimatorOutput::marg_data` is already an owned in-memory value) into a
//! *bounded* (`--mapper-queue-capacity`, generous default) channel to a
//! dedicated mapper thread running `visloc_basalt::mapper_online::
//! OnlineNfrMapper`. The VIO thread only blocks on the mapper once the
//! mapper has fallen behind by a full queue capacity's worth of packets --
//! see `mapper_online::MapperPacketSender`'s doc for why an unbounded
//! channel here let a slow mapper packet under `--pipeline` (whose
//! frontend/estimator overlap makes the VIO producer materially faster than
//! serial) turn into multi-GB backlog growth and a memory-pressure-amplified
//! apparent stall. Backpressure -- how often and how long the VIO thread
//! actually blocks -- is observable and reported as mapper queue depth/lag
//! (`max_mapper_queue_depth`, `mapper_queue_lag_seconds`).
//!
//! See `docs/basalt_online_mapper_design.md` for the full design and
//! `docs/vi_slam_global_consistency_plan.md` Sec1.4/3/4 for why this exists:
//! turning the offline NFR mapper that already reaches 8/11 wins vs
//! ORB-SLAM3 into an online system that runs concurrently with the VIO.
//!
//! ```text
//! cargo run --release --example basalt_euroc_online_slam_demo -- \
//!   --euroc-dir /data/MH_01_easy \
//!   --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
//!   --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
//!   --out-dir target/basalt_mh01_online --optimize-every-k 100 --periodic-iterations 4
//! ```

use std::{
    collections::{BTreeMap, VecDeque},
    env, fs,
    io::{BufWriter, Write},
    path::PathBuf,
    process,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use serde_json::json;
use visloc_basalt::{
    mapper_online::{
        run_mapper_thread, OnlineIngestReport, OnlineMapperConfig, OnlineNfrMapper, SentImageFilter,
    },
    vio::MargData,
    BasaltAdapterError, BasaltAdapterOutput, BasaltVioEstimatorAdapter, EurocSensorDataset,
    RejectReason, TimingBreakdown,
};
use visloc_core::geometry::SE3;

// Opt-in global allocator swap (`--features mimalloc-global`): the VIO
// estimator's LM trial loop allocates a scratch buffer per factor per
// trial, so a faster allocator may help. Pure runtime substitution --
// does not change any algorithm's output.
#[cfg(feature = "mimalloc-global")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug)]
struct Args {
    euroc_dir: PathBuf,
    calibration: PathBuf,
    config: PathBuf,
    out_dir: PathBuf,
    /// Optional per-frame frontend tracking diagnostics CSV.
    frontend_stats_csv: Option<PathBuf>,
    max_frames: Option<usize>,
    optimize_every_k: usize,
    periodic_iterations: usize,
    realtime: bool,
    /// Two-thread frontend/estimator overlap (PR #153). On by default: it is
    /// bit-identical to the serial path (verified: identical trajectory.tum
    /// SHA-256 and identical marg_data/ contents on MH_03/MH_04) and
    /// materially faster, so there is no reason to ship the slower serial
    /// default. `--no-pipeline` restores the serial path.
    pipeline: bool,
    /// True when `--pipeline` or `--no-pipeline` was typed explicitly, so
    /// `--realtime`'s serial-only pacing can silently fall back to serial
    /// when `pipeline` is just the default, while still rejecting an
    /// explicit `--pipeline --realtime` request.
    pipeline_explicit: bool,
    pipeline_capacity: usize,
    decode_threads: usize,
    threads: Option<usize>,
    mapper_queue_capacity: usize,
    no_urgent_keyframes: bool,
    imu_seed_klt: bool,
    /// Restores the legacy diagnostic MargData LM path (per-trial landmark
    /// re-factorization + pre-solve diagnostic linearization).  Off by default:
    /// the compact path is byte-identical in trajectory and MargData bytes, so
    /// it is the canonical online-mapper setting.
    retained_marg_diagnostics: bool,
    /// LM iteration budget for the final full global-BA pass
    /// (`NfrMapperHeadlessConfig::num_opt_iter`).  Defaults to the mapper
    /// contract's 10.
    num_opt_iter: usize,
    /// Loop/temporal-match candidate count per new keyframe query override
    /// (None keeps the mapper default, `OnlineMapperConfig::match_top_k`).
    match_top_k: Option<usize>,
    /// Enable L1 projection-based persistent-landmark re-observation.
    projection_rematch: bool,
    /// Enable incremental local mapping (per-keyframe triangulation into the
    /// live landmark map).
    incremental_local_mapping: bool,
    /// Projection host-rank window override (None keeps the mapper default).
    projection_host_window: Option<u64>,
    /// Projection search radius override in pixels (None keeps the default).
    projection_radius_px: Option<f64>,
    /// Disable insertion of loop matches into the map (diagnostic/robustness
    /// control; default is to insert them).
    no_loop_matching: bool,
    /// Loop-match rotation gate override in degrees (None keeps the default).
    loop_match_max_rotation_error_deg: Option<f64>,
    /// Enable loop-closure relative-pose factors recovered from map landmarks.
    loop_closure_factors: bool,
    /// Minimum 3D-3D correspondences for a loop factor.
    loop_closure_min_correspondences: usize,
    /// Scalar weight for loop-closure factors.
    loop_closure_weight: f64,
    /// Max loop rotation-vs-VIO disagreement in degrees.
    loop_closure_max_rotation_error_deg: f64,
    /// Covisibility local-BA window size in keyframes (0 disables).
    local_ba_window: usize,
    /// LM iterations for the local windowed BA.
    local_ba_iterations: usize,
    /// Scalar multiplier on the marginalisation-derived (IMU-informed)
    /// relative-pose factors the NFR mapper already recovers from each
    /// `MargData` packet (`MapperFactors::relative_pose`,
    /// `pipelines/basalt/src/mapper/mod.rs::extract_nonlinear_factors`).
    /// `None` keeps `MapperConfig::default()`'s weight of `1.0` (unchanged
    /// behavior). These edges carry the VIO window's own covariance
    /// (position + yaw), so raising this weight strengthens the global BA's
    /// metric-scale/drift constraint relative to its vision-only reprojection
    /// and loop-closure terms without adding any per-frame VIO cost.
    relative_pose_weight: Option<f64>,
    /// Weight for raw preintegrated-IMU mapper edges; None means disabled (0.0).
    imu_preintegration_weight: Option<f64>,
    /// Weight for the joint visual-inertial global BA (full 15-dof-per-
    /// keyframe navigation state: pose + velocity + gyro/accel bias, solved
    /// jointly with preintegrated-IMU and bias-random-walk factors between
    /// consecutive keyframes; see `mapper::imu_ba`). A materially different,
    /// more expensive lever than `imu_preintegration_weight`'s frozen-
    /// velocity edges; None means disabled (0.0).
    joint_vi_ba_weight: Option<f64>,
    /// Scalar multiplier on the marginalisation-derived roll/pitch
    /// (gravity-direction) factors (`MapperFactors::roll_pitch`, only
    /// emitted when the source `MargData` used IMU). `None` keeps
    /// `MapperConfig::default()`'s weight of `1.0`.
    roll_pitch_weight: Option<f64>,
}

/// Default bound on the VIO-to-mapper `MargData` channel (see
/// `mapper_online::MapperPacketSender`'s doc). Sized well above the queue
/// depths a keeping-pace mapper reaches in practice (single digits to a few
/// dozen, per `max_mapper_queue_depth` on ordinary runs) so it does not
/// throttle a healthy run, while still capping worst-case backlog -- and
/// therefore worst-case retained-raw-image memory -- to a fixed multiple of
/// one packet's `of_images` payload instead of growing without bound.
const DEFAULT_MAPPER_QUEUE_CAPACITY: usize = 256;

fn main() {
    if let Err(error) = run() {
        eprintln!("basalt_euroc_online_slam_demo: {error}");
        process::exit(1);
    }
}

/// Running sums of [`OnlineIngestReport`] fields across the whole sequence,
/// accumulated on the mapper thread and read back after it joins. Kept as
/// aggregate counters (not one record per packet) so the output stays small
/// even on EuRoC's largest sequences.
#[derive(Debug, Default)]
struct MapperAggregate {
    packet_count: usize,
    new_key_count: usize,
    accepted_temporal_pair_count: usize,
    accepted_loop_pair_count: usize,
    /// A background optimize job was *started* (rule 1's rate-limited
    /// trigger; see `mapper_online`'s module doc).
    optimize_trigger_count: usize,
    /// A background optimize job *finished and was merged* -- may lag
    /// `optimize_trigger_count` by up to one in-flight job at any time.
    optimize_merge_count: usize,
    detect_seconds: f64,
    stereo_seconds: f64,
    match_seconds: f64,
    /// Sum/max of merged background jobs' wall time and per-stage
    /// breakdown, for the "where does the time go" report.
    optimize_total_seconds: f64,
    optimize_max_seconds: f64,
    optimize_build_tracks_seconds: f64,
    optimize_setup_opt_seconds: f64,
    optimize_lm_seconds: f64,
    optimize_filter_seconds: f64,
}

impl MapperAggregate {
    fn add(&mut self, report: &OnlineIngestReport) {
        self.packet_count += 1;
        self.new_key_count += report.new_key_count;
        self.accepted_temporal_pair_count += report.accepted_temporal_pair_count;
        self.accepted_loop_pair_count += report.accepted_loop_pair_count;
        self.detect_seconds += report.detect_seconds;
        self.stereo_seconds += report.stereo_seconds;
        self.match_seconds += report.match_seconds;
        if report.optimize_triggered {
            self.optimize_trigger_count += 1;
        }
        if let Some(breakdown) = report.optimize_merge {
            self.optimize_merge_count += 1;
            self.optimize_total_seconds += breakdown.total_seconds;
            self.optimize_max_seconds = self.optimize_max_seconds.max(breakdown.total_seconds);
            self.optimize_build_tracks_seconds += breakdown.build_tracks_seconds;
            self.optimize_setup_opt_seconds += breakdown.setup_opt_seconds;
            self.optimize_lm_seconds += breakdown.optimize1_seconds + breakdown.optimize2_seconds;
            self.optimize_filter_seconds += breakdown.filter_seconds;
        }
    }
}

/// Resolves the `--pipeline`/`--realtime` interaction: `--realtime`'s
/// dataset-rate pacing only exists on the serial path (`--pipeline`'s
/// frontend/estimator overlap is incompatible with per-frame pacing by
/// design). Since `--pipeline` defaults on, an unqualified `--realtime`
/// silently falls back to serial instead of erroring on a flag combination
/// the caller never typed; an *explicit* `--pipeline --realtime` request is
/// still rejected rather than silently downgraded.
fn resolve_pipeline_realtime(
    pipeline: bool,
    pipeline_explicit: bool,
    realtime: bool,
) -> Result<bool, &'static str> {
    if !(pipeline && realtime) {
        return Ok(pipeline);
    }
    if pipeline_explicit {
        return Err("--realtime is not supported together with --pipeline");
    }
    Ok(false)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse(env::args_os().skip(1))
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    args.pipeline =
        resolve_pipeline_realtime(args.pipeline, args.pipeline_explicit, args.realtime)?;
    // Sizes the process-wide rayon pool used by data-parallel stages inside
    // the adapter/estimator (PR #153). Same flag/behavior as
    // examples/basalt_euroc_vio_demo.rs's `--threads`.
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("failed to configure rayon thread pool: {error}"))?;
    }
    fs::create_dir_all(&args.out_dir)?;

    let dataset = EurocSensorDataset::open(&args.euroc_dir, &args.calibration, &args.config)?;
    // Urgent keyframe spacing is on by default in this demo: a keyframe may be
    // taken two frames after the last one when fewer than half of the tracks
    // are connected. On 11 EuRoC sequences it improved ATE on 7 sequences
    // (V2_03 −17%, MH_04 −16%), left 3 unchanged and cost MH_02 +5%. Config
    // values win; `--no-urgent-keyframes` restores the upstream decision.
    let mut vio_config = dataset.config().clone();
    if args.imu_seed_klt {
        vio_config.values.insert(
            "config.optical_flow_imu_seed_rotation".to_string(),
            json!(true),
        );
    }
    if !args.no_urgent_keyframes {
        for (key, value) in [
            ("config.vio_urgent_kf_keypoints_thresh", json!(0.5)),
            ("config.vio_urgent_min_frames_after_kf", json!(2)),
        ] {
            vio_config.values.entry(key.to_string()).or_insert(value);
        }
    }
    let mut adapter = BasaltVioEstimatorAdapter::from_config(dataset.calibration(), &vio_config)?;
    // Default: compact MargData LM path.  Trajectory and MargData bytes are
    // identical to the diagnostic path (verified by the retained-vs-lean
    // window regression test); only the per-trial diagnostic trace is skipped.
    adapter
        .estimator
        .set_lean_marg_data(!args.retained_marg_diagnostics);
    let frame_limit = args
        .max_frames
        .unwrap_or(dataset.frame_count())
        .min(dataset.frame_count());
    if frame_limit == 0 {
        return Err("EuRoC cam0 manifest has no frames".into());
    }

    let mut mapper_config = dataset.config().mapper_config()?;
    if let Some(weight) = args.relative_pose_weight {
        mapper_config.relative_pose_weight = weight;
    }
    if let Some(weight) = args.roll_pitch_weight {
        mapper_config.roll_pitch_weight = weight;
    }
    let feature_config = dataset.config().offline_mapper_config()?;
    let optimize_config = dataset.config().mapper_global_ba_config()?;
    let headless = visloc_basalt::mapper::NfrMapperHeadlessConfig {
        num_opt_iter: args.num_opt_iter,
        ..visloc_basalt::mapper::NfrMapperHeadlessConfig::default()
    };
    let imu_samples_for_mapper =
        if args.imu_preintegration_weight.is_some() || args.joint_vi_ba_weight.is_some() {
            std::sync::Arc::from(dataset.imu_samples())
        } else {
            std::sync::Arc::from([])
        };
    let online_mapper = OnlineNfrMapper::new(
        mapper_config,
        dataset.calibration().clone(),
        feature_config,
        optimize_config,
        OnlineMapperConfig {
            imu_samples: imu_samples_for_mapper,
            imu_preintegration_weight: args.imu_preintegration_weight.unwrap_or(0.0),
            joint_vi_ba_weight: args.joint_vi_ba_weight.unwrap_or(0.0),
            optimize_every_k: args.optimize_every_k,
            periodic_iterations: args.periodic_iterations,
            headless,
            match_top_k: args
                .match_top_k
                .unwrap_or(OnlineMapperConfig::default().match_top_k),
            projection_rematch: args.projection_rematch,
            incremental_local_mapping: args.incremental_local_mapping,
            projection_host_window: args
                .projection_host_window
                .unwrap_or(OnlineMapperConfig::default().projection_host_window),
            projection_search_radius_px: args
                .projection_radius_px
                .unwrap_or(OnlineMapperConfig::default().projection_search_radius_px),
            loop_matching: !args.no_loop_matching,
            loop_match_max_rotation_error_deg: args
                .loop_match_max_rotation_error_deg
                .unwrap_or(OnlineMapperConfig::default().loop_match_max_rotation_error_deg),
            loop_closure_factors: args.loop_closure_factors,
            loop_closure_min_correspondences: args.loop_closure_min_correspondences,
            loop_closure_weight: args.loop_closure_weight,
            loop_closure_max_rotation_error_deg: args.loop_closure_max_rotation_error_deg,
            local_ba_window: args.local_ba_window,
            local_ba_iterations: args.local_ba_iterations,
            ..OnlineMapperConfig::default()
        },
    );

    // Bounded: see `mapper_online::MapperPacketSender`'s doc for why an
    // unbounded channel let a slow mapper packet turn into unbounded
    // backlog growth (and, on a loaded host, memory-pressure-amplified
    // apparent stalls) specifically under `--pipeline`, whose
    // frontend/estimator overlap makes the VIO producer materially faster
    // than the mapper's own per-packet consumption rate. The bound is sized
    // generously (`--mapper-queue-capacity`, default below) so ordinary
    // operation -- where the mapper keeps up within a small multiple of a
    // packet's own processing time -- never blocks the VIO thread; it only
    // throttles VIO once the mapper has genuinely fallen far behind, which
    // is exactly the condition that must be bounded rather than left to
    // grow without limit. Backpressure is still measured and reported
    // (queue depth, lag) so a run that spends real time blocked here is
    // visible in the summary, not just implicitly slower.
    let (sender, receiver) = mpsc::sync_channel::<MargData>(args.mapper_queue_capacity);
    let send_times: Arc<Mutex<VecDeque<Instant>>> = Arc::new(Mutex::new(VecDeque::new()));
    let lag_seconds: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let aggregate: Arc<Mutex<MapperAggregate>> = Arc::new(Mutex::new(MapperAggregate::default()));

    let send_times_for_mapper = Arc::clone(&send_times);
    let lag_seconds_for_mapper = Arc::clone(&lag_seconds);
    let aggregate_for_mapper = Arc::clone(&aggregate);
    let mapper_handle = thread::spawn(move || {
        let mapper_processed = std::sync::atomic::AtomicU64::new(0);
        run_mapper_thread(online_mapper, receiver, None, move |report| {
            if let Some(sent_at) = send_times_for_mapper.lock().expect("lock").pop_front() {
                lag_seconds_for_mapper
                    .lock()
                    .expect("lock")
                    .push(sent_at.elapsed().as_secs_f64());
            }
            aggregate_for_mapper.lock().expect("lock").add(report);
            let processed = mapper_processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            // Live progress: this is the only visibility into the mapper
            // thread once the VIO thread's own frame-count log has finished
            // (mapper packets can queue up behind a slow keyframe, and a
            // periodic optimize's own cost only shows up here). Print every
            // packet during the (typically short) tail so a queue drain or
            // an unexpectedly slow optimize pass is visible live rather than
            // only in the final aggregate JSON.
            let merge_note = match report.optimize_merge {
                Some(breakdown) => format!(
                    " MERGED(total={:.2}s build_tracks={:.2}s setup_opt={:.2}s lm={:.2}s filter={:.2}s)",
                    breakdown.total_seconds,
                    breakdown.build_tracks_seconds,
                    breakdown.setup_opt_seconds,
                    breakdown.optimize1_seconds + breakdown.optimize2_seconds,
                    breakdown.filter_seconds,
                ),
                None => String::new(),
            };
            eprintln!(
                "mapper packet={processed} new_keys={} detect={:.2}s stereo={:.2}s match={:.2}s \
                 optimize_triggered={} accepted_loops={}{merge_note}",
                report.new_key_count,
                report.detect_seconds,
                report.stereo_seconds,
                report.match_seconds,
                report.optimize_triggered,
                report.accepted_loop_pair_count,
            );
        })
    });

    let mut first_timestamp_ns: Option<i64> = None;
    let mut last_timestamp_ns: i64 = 0;
    let mut total_imu = 0usize;
    let mut total_observations = 0usize;
    let mut mapper_packet_count = 0u64;
    let mut sent_images = SentImageFilter::new();
    let mut max_queue_depth = 0usize;
    let mut demo_index = 0usize;
    // Every processed frame's raw VIO body-to-world pose, keyed by frame_id.
    // Used after the run to propagate the mapper's keyframe corrections to
    // every frame (see `propagate_to_all_frames` below) -- the same rigid
    // spanning-tree convention as scripts/propagate_basalt_mapper_corrections.py.
    let mut vio_trajectory: BTreeMap<u64, (i64, SE3)> = BTreeMap::new();
    let mut frontend_stats_csv = args
        .frontend_stats_csv
        .as_ref()
        .map(|path| fs::File::create(path).map(BufWriter::new))
        .transpose()?;
    let vio_start = Instant::now();

    // Shared per-frame handler for both the serial loop and the `--pipeline`
    // two-thread path below -- identical to how
    // examples/basalt_euroc_vio_demo.rs shares one `handle_output` between
    // its own serial and pipelined call sites. This closure only reads
    // `output` (already fully computed by the adapter) and updates this
    // demo's own bookkeeping/channel-send; it never touches VIO arithmetic,
    // which is what keeps `--pipeline`'s output identical to the serial
    // path's.
    let mut handle_output = |output: BasaltAdapterOutput,
                             _timing: &mut TimingBreakdown|
     -> Result<(), BasaltAdapterError> {
        let timestamp_ns = output.tracks.frame.timestamp_ns;
        if last_timestamp_ns != 0 && timestamp_ns <= last_timestamp_ns {
            return Err(BasaltAdapterError::Output(format!(
                "non-monotonic output timestamp at frame {demo_index}"
            )));
        }
        first_timestamp_ns.get_or_insert(timestamp_ns);
        last_timestamp_ns = timestamp_ns;
        total_imu += output.imu_count;
        total_observations += output.tracks.observations.len();
        vio_trajectory.insert(
            output.tracks.frame.frame_id,
            (timestamp_ns, output.estimator.state.imu_to_world.clone()),
        );

        if output.estimator.marg_data.is_mapper_packet() {
            let mut marg_data = output.estimator.marg_data;
            sent_images.strip_sent(&mut marg_data);
            let depth_after_send = {
                let mut guard = send_times.lock().expect("lock");
                guard.push_back(Instant::now());
                guard.len()
            };
            max_queue_depth = max_queue_depth.max(depth_after_send);
            sender.send(marg_data).map_err(|_| {
                BasaltAdapterError::Output(
                    "mapper thread ended before the VIO stream finished".into(),
                )
            })?;
            mapper_packet_count += 1;
        }

        if demo_index == 0 || (demo_index + 1).is_multiple_of(50) || demo_index + 1 == frame_limit {
            eprintln!(
                "frame={} timestamp_ns={} mapper_packets={} imu={}",
                output.tracks.frame.frame_id, timestamp_ns, mapper_packet_count, total_imu,
            );
        }
        if let Some(writer) = frontend_stats_csv.as_mut() {
            if demo_index == 0 {
                writeln!(
                    writer,
                    "frame_index,frame_id,timestamp_ns,num_observations,num_created,num_retained,num_rejected,\
                     reject_frame_forward,reject_frame_backward,reject_frame_fb_squared,\
                     reject_existing_stereo_forward,reject_existing_stereo_backward,reject_existing_stereo_fb_squared,\
                     reject_fast_no_candidate,reject_stereo_forward,reject_stereo_backward,\
                     reject_stereo_fb_squared,reject_stereo_bearing_invalid,reject_stereo_essential_residual"
                )
                .map_err(|error| BasaltAdapterError::Output(error.to_string()))?;
            }
            let mut frame_forward = 0usize;
            let mut frame_backward = 0usize;
            let mut frame_fb_squared = 0usize;
            let mut existing_stereo_forward = 0usize;
            let mut existing_stereo_backward = 0usize;
            let mut existing_stereo_fb_squared = 0usize;
            let mut fast_no_candidate = 0usize;
            let mut stereo_forward = 0usize;
            let mut stereo_backward = 0usize;
            let mut stereo_fb_squared = 0usize;
            let mut stereo_bearing_invalid = 0usize;
            let mut stereo_essential_residual = 0usize;
            for (reason, count) in output.tracks.reject_counters.iter() {
                let bucket = match reason {
                    RejectReason::FrameForward(..) => &mut frame_forward,
                    RejectReason::FrameBackward(..) => &mut frame_backward,
                    RejectReason::FrameFbSquared => &mut frame_fb_squared,
                    RejectReason::ExistingStereoForward(..) => &mut existing_stereo_forward,
                    RejectReason::ExistingStereoBackward(..) => &mut existing_stereo_backward,
                    RejectReason::ExistingStereoFbSquared => &mut existing_stereo_fb_squared,
                    RejectReason::FastNoCandidate => &mut fast_no_candidate,
                    // The opt-in reprojection seed rejects a keypoint before
                    // the forward search; count it with the forward failures
                    // so the CSV layout stays unchanged.
                    RejectReason::StereoForward(..) | RejectReason::StereoSeedOutOfView => {
                        &mut stereo_forward
                    }
                    RejectReason::StereoBackward(..) => &mut stereo_backward,
                    RejectReason::StereoFbSquared => &mut stereo_fb_squared,
                    RejectReason::StereoBearingInvalid => &mut stereo_bearing_invalid,
                    RejectReason::StereoEssentialResidual => &mut stereo_essential_residual,
                };
                *bucket += *count;
            }
            writeln!(
                writer,
                "{demo_index},{},{timestamp_ns},{},{},{},{},\
                 {frame_forward},{frame_backward},{frame_fb_squared},\
                 {existing_stereo_forward},{existing_stereo_backward},{existing_stereo_fb_squared},\
                 {fast_no_candidate},{stereo_forward},{stereo_backward},\
                 {stereo_fb_squared},{stereo_bearing_invalid},{stereo_essential_residual}",
                output.tracks.frame.frame_id,
                output.tracks.observations.len(),
                output.tracks.created_track_ids.len(),
                output.tracks.retained_track_ids.len(),
                output.tracks.rejected_track_ids.len(),
            )
            .map_err(|error| BasaltAdapterError::Output(error.to_string()))?;
        }
        demo_index += 1;
        Ok(())
    };

    if args.pipeline {
        // Two-thread pipeline: a frontend/producer thread (dataset
        // acquisition + tracking) overlapped in wall time with the
        // estimator/consumer thread running on this thread (PR #153). Frame
        // order and every per-frame computation are unchanged from the
        // serial path below; this demo supplies the same `handle_output`
        // either way.
        let (_producer_timing, _consumer_timing) = adapter.process_euroc_stream_pipelined(
            &dataset,
            frame_limit,
            true,  // retain_marg_data: this demo always needs mapper packets
            false, // retain_trace: this demo has no --no-trace-style trace output
            args.pipeline_capacity,
            args.decode_threads,
            handle_output,
        )?;
    } else {
        let mut timing = TimingBreakdown::from_env();
        // Tracked separately from the closure-captured `first_timestamp_ns`
        // (a borrow of which would otherwise have to outlive the closure
        // itself) -- pacing only needs the first frame's own timestamp,
        // read directly off the dataset before `handle_output` runs.
        let mut pacing_first_timestamp_ns: Option<i64> = None;
        for index in 0..frame_limit {
            let sensor_frame = dataset.frame(index)?;
            if args.realtime {
                let first = *pacing_first_timestamp_ns.get_or_insert(sensor_frame.timestamp_ns);
                let target_offset =
                    Duration::from_nanos((sensor_frame.timestamp_ns - first).max(0) as u64);
                let target = vio_start + target_offset;
                let now = Instant::now();
                if target > now {
                    thread::sleep(target - now);
                }
            }
            let output = adapter.process(sensor_frame)?;
            handle_output(output, &mut timing)?;
        }
    }
    let vio_wall_seconds = vio_start.elapsed().as_secs_f64();

    // Close the channel: the mapper thread's receive loop exits and this
    // demo runs OnlineNfrMapper::finalize (rule 3 -- the exact
    // `run_headless` tail) after joining, below.
    drop(sender);
    let (mut online_mapper, _stop_reason, mapper_errors) =
        mapper_handle.join().map_err(|_| "mapper thread panicked")?;
    if let Some(error) = mapper_errors.first() {
        return Err(format!("mapper thread reported an error: {error}").into());
    }
    let mapper_join_seconds = vio_start.elapsed().as_secs_f64() - vio_wall_seconds;

    let mapper_heap = online_mapper.inner().approx_heap_breakdown();
    let final_report = online_mapper.finalize()?;
    eprintln!(
        "mapper_heap_estimate {}",
        mapper_heap
            .iter()
            .map(|(name, bytes)| format!("{name}={bytes}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let total_wall_seconds = vio_start.elapsed().as_secs_f64();
    eprintln!(
        "mapper_psd_information_projections={}",
        visloc_basalt::mapper::psd_information_projection_count()
    );

    // Propagate the mapper's keyframe corrections to every VIO frame (rigid
    // spanning-tree: scripts/propagate_basalt_mapper_corrections.py's exact
    // convention, ported to Rust). `final_report.trajectory_tum` remains the
    // keyframe-only trajectory (written separately below); this is the
    // full-frame trajectory the sweep evaluates, matching the offline
    // path's `trajectory_full_propagated.tum` protocol.
    let mapper_poses: BTreeMap<u64, SE3> = final_report
        .result
        .poses
        .iter()
        .map(|record| {
            let [qw, qx, qy, qz] = record.quaternion_wxyz;
            let pose = SE3::new(
                nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    qw, qx, qy, qz,
                )),
                nalgebra::Vector3::new(
                    record.translation[0],
                    record.translation[1],
                    record.translation[2],
                ),
            );
            (record.frame_id, pose)
        })
        .collect();
    let propagated = propagate_to_all_frames(&vio_trajectory, &mapper_poses)?;
    let full_trajectory_tum = tum_string(&propagated);
    let interpolated = propagate_interpolated(&vio_trajectory, &mapper_poses)?;
    let interpolated_trajectory_tum = tum_string(&interpolated);
    let raw_vio: Vec<(i64, SE3)> = vio_trajectory
        .values()
        .map(|(timestamp_ns, pose)| (*timestamp_ns, pose.clone()))
        .collect();
    let vio_trajectory_tum = tum_string(&raw_vio);
    let propagated_frame_count = propagated.len();

    let dataset_duration_seconds =
        (last_timestamp_ns - first_timestamp_ns.unwrap_or(last_timestamp_ns)) as f64 * 1e-9;
    let real_time_factor = if vio_wall_seconds > 0.0 {
        dataset_duration_seconds / vio_wall_seconds
    } else {
        0.0
    };

    let lag = lag_seconds.lock().expect("lock");
    let lag_max = lag.iter().cloned().fold(0.0_f64, f64::max);
    let lag_sample_count = lag.len();
    let lag_mean = if lag.is_empty() {
        0.0
    } else {
        lag.iter().sum::<f64>() / lag.len() as f64
    };
    drop(lag);
    let aggregate = aggregate.lock().expect("lock");
    let peak_rss_bytes = peak_working_set_bytes();

    let summary = json!({
        "schema": "basalt.online_mapper.run.v1",
        "lean_marg_data": !args.retained_marg_diagnostics,
        "final_optimize_iterations_budget": args.num_opt_iter,
        "projection_rematch": args.projection_rematch,
        "incremental_local_mapping": args.incremental_local_mapping,
        "pacing": if args.realtime { "dataset_rate" } else { "as_fast_as_possible" },
        "frames_processed": frame_limit,
        "dataset_duration_seconds": dataset_duration_seconds,
        "vio_wall_seconds": vio_wall_seconds,
        "mapper_join_seconds": mapper_join_seconds,
        "total_wall_seconds": total_wall_seconds,
        "real_time_factor": real_time_factor,
        "imu_samples": total_imu,
        "observations": total_observations,
        "mapper_packets_sent": mapper_packet_count,
        "mapper_queue_capacity": args.mapper_queue_capacity,
        "max_mapper_queue_depth": max_queue_depth,
        "mapper_queue_lag_seconds": {"max": lag_max, "mean": lag_mean, "samples": lag_sample_count},
        "mapper": {
            "packet_count": aggregate.packet_count,
            "new_key_count": aggregate.new_key_count,
            "accepted_temporal_pair_count": aggregate.accepted_temporal_pair_count,
            "accepted_loop_pair_count": aggregate.accepted_loop_pair_count,
            "optimize_trigger_count": aggregate.optimize_trigger_count,
            "optimize_merge_count": aggregate.optimize_merge_count,
            "detect_seconds": aggregate.detect_seconds,
            "stereo_seconds": aggregate.stereo_seconds,
            "match_seconds": aggregate.match_seconds,
            "optimize_total_seconds": aggregate.optimize_total_seconds,
            "optimize_max_seconds": aggregate.optimize_max_seconds,
            "optimize_build_tracks_seconds": aggregate.optimize_build_tracks_seconds,
            "optimize_setup_opt_seconds": aggregate.optimize_setup_opt_seconds,
            "optimize_lm_seconds": aggregate.optimize_lm_seconds,
            "optimize_filter_seconds": aggregate.optimize_filter_seconds,
        },
        "final_optimize": {
            "first_optimize_initial_cost": final_report.first_optimize.initial_cost,
            "first_optimize_final_cost": final_report.first_optimize.final_cost,
            "first_optimize_iterations": final_report.first_optimize.iterations,
            "first_optimize_requested_iterations": final_report.first_optimize.requested_iterations,
            "first_optimize_accepted_steps": final_report.first_optimize.accepted_step_count,
            "first_optimize_rejected_trials": final_report.first_optimize.rejected_trial_count,
            "first_optimize_initial_lambda": final_report.first_optimize.initial_lambda,
            "first_optimize_trials": final_report
                .first_optimize
                .trace
                .iter()
                .flat_map(|iteration| {
                    iteration.trials.iter().map(move |trial| {
                        json!({
                            "iteration": iteration.iteration,
                            "total_cost": iteration.total_cost,
                            "lambda": trial.lambda,
                            "f_diff": trial.f_diff,
                            "max_pose_increment": trial.max_pose_increment,
                            "after_vision": trial.after_vision_cost,
                            "after_relative": trial.after_relative_cost,
                            "after_roll_pitch": trial.after_roll_pitch_cost,
                            "accepted": trial.accepted,
                        })
                    })
                })
                .collect::<Vec<_>>(),
            "second_optimize_final_cost": final_report.second_optimize.final_cost,
            "second_optimize_iterations": final_report.second_optimize.iterations,
            "second_optimize_rejected_trials": final_report.second_optimize.rejected_trial_count,
            "pose_count": final_report.result.poses.len(),
            "landmark_count": final_report.result.landmarks.len(),
            "filter_before_landmarks": final_report.filter.before_landmark_count,
            "filter_after_landmarks": final_report.filter.after_landmark_count,
            "filter_reprojection_error": final_report.filter.reprojection_error,
        },
        "propagated_frame_count": propagated_frame_count,
        "peak_working_set_bytes": peak_rss_bytes,
    });

    // `trajectory_online.tum` is the full-frame propagated trajectory (every
    // VIO frame, keyframe corrections applied) -- the file the evaluation
    // sweep scores, matching the offline path's protocol.
    // `trajectory_online_kf.tum` is the mapper's own keyframe-only output,
    // kept for debugging/comparison.
    // `trajectory_online.tum` interpolates the keyframe corrections between
    // the surrounding keyframes (`propagate_interpolated`). On six EuRoC
    // sequences this cut consecutive RPE by up to 60% with equal or better
    // ATE compared with the piecewise-constant propagation, which is kept as
    // `trajectory_online_nearest.tum`. `trajectory_vio.tum` is the
    // uncorrected VIO, for reference.
    if let Some(writer) = frontend_stats_csv.as_mut() {
        writer.flush()?;
    }
    fs::write(
        args.out_dir.join("trajectory_online.tum"),
        &interpolated_trajectory_tum,
    )?;
    fs::write(
        args.out_dir.join("trajectory_online_nearest.tum"),
        &full_trajectory_tum,
    )?;
    fs::write(args.out_dir.join("trajectory_vio.tum"), &vio_trajectory_tum)?;
    fs::write(
        args.out_dir.join("trajectory_online_kf.tum"),
        &final_report.trajectory_tum,
    )?;
    fs::write(
        args.out_dir.join("timing_breakdown_online.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    eprintln!(
        "pacing={} rtf={:.3} vio_wall={:.1}s total_wall={:.1}s max_queue_depth={} \
         lag_max={:.3}s lag_mean={:.3}s loops={} triggers={} merges={} \
         optimize_max={:.1}s optimize_total={:.1}s peak_rss={:.0}MB out={}",
        if args.realtime {
            "dataset_rate"
        } else {
            "as_fast_as_possible"
        },
        real_time_factor,
        vio_wall_seconds,
        total_wall_seconds,
        max_queue_depth,
        lag_max,
        lag_mean,
        aggregate.accepted_loop_pair_count,
        aggregate.optimize_trigger_count,
        aggregate.optimize_merge_count,
        aggregate.optimize_max_seconds,
        aggregate.optimize_total_seconds,
        peak_rss_bytes as f64 / 1e6,
        args.out_dir.display(),
    );
    Ok(())
}

#[cfg(windows)]
fn peak_working_set_bytes() -> u64 {
    // Same minimal FFI as examples/basalt_mapper_stage_profile.rs; examples
    // are independent crate roots so this small helper is duplicated rather
    // than shared.
    #[repr(C)]
    #[allow(non_snake_case)]
    struct ProcessMemoryCounters {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
    }
    unsafe {
        let mut counters: ProcessMemoryCounters = std::mem::zeroed();
        counters.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
        let handle = GetCurrentProcess();
        if GetProcessMemoryInfo(handle, &mut counters, counters.cb) != 0 {
            counters.PeakWorkingSetSize as u64
        } else {
            0
        }
    }
}

#[cfg(target_os = "linux")]
fn peak_working_set_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")?
                    .trim()
                    .strip_suffix("kB")?
                    .trim()
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
        })
        .unwrap_or(0)
}

#[cfg(not(any(windows, target_os = "linux")))]
const fn peak_working_set_bytes() -> u64 {
    0
}

/// Rigid spanning-tree propagation of the mapper's keyframe corrections to
/// every VIO frame. A direct Rust port of
/// `scripts/propagate_basalt_mapper_corrections.py::propagate` (kept
/// numerically equivalent, not merely similar -- see that script's
/// module docstring for the exact convention):
///
/// ```text
/// V_f  = raw VIO body-to-world pose at frame f (rotation R_f, translation t_f)
/// M_k  = mapper-corrected body-to-world pose at the nearest preceding
///        keyframe k (frame_id <= f)
/// V_k  = raw VIO body-to-world pose at that same keyframe k
///
/// corrected(f) = Delta_k * V_f,  where Delta_k = M_k * V_k^{-1}
/// ```
///
/// i.e. each frame keeps its VIO-derived relative motion to the nearest
/// preceding keyframe; only the keyframe's mapper correction moves it. This
/// is exact at keyframes themselves (f == k, `Delta_k * V_k == M_k`) and,
/// like the Python original, uses the *first* keyframe's delta for any
/// frame before it (there is no earlier keyframe to interpolate from).
fn propagate_to_all_frames(
    vio_trajectory: &BTreeMap<u64, (i64, SE3)>,
    mapper_poses: &BTreeMap<u64, SE3>,
) -> Result<Vec<(i64, SE3)>, Box<dyn std::error::Error>> {
    let keyframe_ids: Vec<u64> = mapper_poses.keys().copied().collect();
    if keyframe_ids.is_empty() {
        return Err("mapper produced no keyframe poses".into());
    }
    let missing: Vec<u64> = keyframe_ids
        .iter()
        .copied()
        .filter(|k| !vio_trajectory.contains_key(k))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{} mapper keyframe frame_ids are absent from the VIO trajectory (first few: {:?})",
            missing.len(),
            &missing[..missing.len().min(5)]
        )
        .into());
    }

    let delta_by_keyframe: BTreeMap<u64, SE3> = keyframe_ids
        .iter()
        .map(|&k| {
            let (_, v_k) = &vio_trajectory[&k];
            let m_k = &mapper_poses[&k];
            (k, m_k.compose(&v_k.inverse()))
        })
        .collect();

    let mut output = Vec::with_capacity(vio_trajectory.len());
    let mut cursor = 0usize;
    for (&frame_id, (timestamp_ns, v_f)) in vio_trajectory {
        while cursor + 1 < keyframe_ids.len() && keyframe_ids[cursor + 1] <= frame_id {
            cursor += 1;
        }
        let delta = &delta_by_keyframe[&keyframe_ids[cursor]];
        output.push((*timestamp_ns, delta.compose(v_f)));
    }
    Ok(output)
}

fn tum_string(poses: &[(i64, SE3)]) -> String {
    let mut buffer = String::from("# timestamp tx ty tz qx qy qz qw\n");
    for (timestamp_ns, pose) in poses {
        let q = pose.rotation.quaternion();
        buffer.push_str(&format!(
            "{:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e}\n",
            *timestamp_ns as f64 * 1e-9,
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            q.i,
            q.j,
            q.k,
            q.w,
        ));
    }
    buffer
}

/// Like [`propagate_to_all_frames`], but the correction applied to a frame
/// between keyframes `k0 < f < k1` is interpolated on SE(3) by timestamp:
///
/// ```text
/// Delta_f = Delta_k0 * Exp(alpha * Log(Delta_k0^{-1} * Delta_k1)),
/// alpha   = (t_f - t_k0) / (t_k1 - t_k0)
/// corrected(f) = Delta_f * V_f
/// ```
///
/// The piecewise-constant scheme applies `Delta_k0` all the way up to `k1`,
/// then jumps to `Delta_k1`. That discontinuity sits at every keyframe
/// boundary and shows up as relative-pose error. Keyframes themselves are
/// still exact, and frames before the first or after the last keyframe use
/// that keyframe's correction.
fn propagate_interpolated(
    vio_trajectory: &BTreeMap<u64, (i64, SE3)>,
    mapper_poses: &BTreeMap<u64, SE3>,
) -> Result<Vec<(i64, SE3)>, Box<dyn std::error::Error>> {
    // Reuse the validation of the piecewise-constant propagation.
    propagate_to_all_frames(vio_trajectory, mapper_poses)?;
    let keyframes: Vec<(u64, i64, SE3)> = mapper_poses
        .iter()
        .map(|(&k, m_k)| {
            let (t_k, v_k) = &vio_trajectory[&k];
            (k, *t_k, m_k.compose(&v_k.inverse()))
        })
        .collect();
    let mut output = Vec::with_capacity(vio_trajectory.len());
    let mut cursor = 0usize;
    for (&frame_id, (timestamp_ns, v_f)) in vio_trajectory {
        while cursor + 1 < keyframes.len() && keyframes[cursor + 1].0 <= frame_id {
            cursor += 1;
        }
        let (k0, t0, delta0) = &keyframes[cursor];
        let delta = match keyframes.get(cursor + 1) {
            Some((_, t1, delta1)) if frame_id > *k0 && t1 > t0 => {
                let alpha = (*timestamp_ns - t0) as f64 / (t1 - t0) as f64;
                let step = delta0.inverse().compose(delta1).log() * alpha;
                delta0.compose(&SE3::exp(&step))
            }
            _ => delta0.clone(),
        };
        output.push((*timestamp_ns, delta.compose(v_f)));
    }
    Ok(output)
}

impl Args {
    fn parse<I>(arguments: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = std::ffi::OsString>,
    {
        let mut euroc_dir = None;
        let mut calibration = None;
        let mut config = PathBuf::from("configs/basalt/euroc_config.json");
        let mut out_dir = PathBuf::from("target/basalt_euroc_online_slam_demo");
        let mut frontend_stats_csv = None;
        let mut max_frames = None;
        let mut optimize_every_k = OnlineMapperConfig::default().optimize_every_k;
        let mut periodic_iterations = OnlineMapperConfig::default().periodic_iterations;
        let mut realtime = false;
        // Same flags/defaults as examples/basalt_euroc_vio_demo.rs's
        // `--pipeline`/`--pipeline-capacity`/`--decode-threads`/`--threads`
        // (PR #153): a bit-identical two-thread frontend/estimator overlap,
        // reused unmodified via `BasaltVioEstimatorAdapter::
        // process_euroc_stream_pipelined` below -- this demo only supplies
        // the same `on_output` callback the serial path already used.
        let mut pipeline = true;
        let mut pipeline_explicit = false;
        let mut pipeline_capacity = 4usize;
        let mut decode_threads = 3usize;
        let mut threads = None;
        let mut mapper_queue_capacity = DEFAULT_MAPPER_QUEUE_CAPACITY;
        let mut no_urgent_keyframes = false;
        let mut imu_seed_klt = false;
        let mut retained_marg_diagnostics = false;
        let mut num_opt_iter = 10usize;
        let mut match_top_k = None;
        let mut projection_rematch = false;
        let mut incremental_local_mapping = false;
        let mut projection_host_window = None;
        let mut projection_radius_px = None;
        let mut no_loop_matching = false;
        let mut loop_match_max_rotation_error_deg = None;
        let mut loop_closure_factors = false;
        let mut loop_closure_min_correspondences =
            OnlineMapperConfig::default().loop_closure_min_correspondences;
        let mut loop_closure_weight = OnlineMapperConfig::default().loop_closure_weight;
        let mut loop_closure_max_rotation_error_deg =
            OnlineMapperConfig::default().loop_closure_max_rotation_error_deg;
        let mut local_ba_window = OnlineMapperConfig::default().local_ba_window;
        let mut local_ba_iterations = OnlineMapperConfig::default().local_ba_iterations;
        let mut imu_preintegration_weight = None;
        let mut joint_vi_ba_weight = None;
        let mut relative_pose_weight = None;
        let mut roll_pitch_weight = None;
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let option = argument.to_string_lossy().into_owned();
            match option.as_str() {
                "--help" | "-h" => return Err(Self::usage()),
                "--euroc-dir" => euroc_dir = Some(PathBuf::from(next(&mut arguments, &option)?)),
                "--calibration" => {
                    calibration = Some(PathBuf::from(next(&mut arguments, &option)?))
                }
                "--config" => config = PathBuf::from(next(&mut arguments, &option)?),
                "--out-dir" => out_dir = PathBuf::from(next(&mut arguments, &option)?),
                "--frontend-stats-csv" => {
                    frontend_stats_csv = Some(PathBuf::from(next(&mut arguments, &option)?))
                }
                "--max-frames" => {
                    max_frames = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<usize>()
                            .map_err(|error| format!("invalid --max-frames: {error}"))?,
                    );
                }
                "--optimize-every-k" => {
                    optimize_every_k = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --optimize-every-k: {error}"))?;
                }
                "--periodic-iterations" => {
                    periodic_iterations = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --periodic-iterations: {error}"))?;
                }
                "--realtime" => realtime = true,
                "--as-fast-as-possible" => realtime = false,
                "--pipeline" => {
                    pipeline = true;
                    pipeline_explicit = true;
                }
                "--no-pipeline" => {
                    pipeline = false;
                    pipeline_explicit = true;
                }
                "--no-urgent-keyframes" => no_urgent_keyframes = true,
                "--imu-seed-klt" => imu_seed_klt = true,
                "--pipeline-capacity" => {
                    pipeline_capacity = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --pipeline-capacity: {error}"))?;
                    if pipeline_capacity == 0 {
                        return Err("--pipeline-capacity must be positive".into());
                    }
                }
                "--decode-threads" => {
                    decode_threads = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --decode-threads: {error}"))?;
                }
                "--threads" => {
                    let value = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --threads: {error}"))?;
                    if value == 0 {
                        return Err("--threads must be positive".into());
                    }
                    threads = Some(value);
                }
                "--mapper-queue-capacity" => {
                    mapper_queue_capacity = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --mapper-queue-capacity: {error}"))?;
                    if mapper_queue_capacity == 0 {
                        return Err("--mapper-queue-capacity must be positive".into());
                    }
                }
                "--retained-marg-diagnostics" => retained_marg_diagnostics = true,
                "--projection-rematch" => projection_rematch = true,
                "--local-mapping" => incremental_local_mapping = true,
                "--loop-closure-max-rot-error" => {
                    loop_closure_max_rotation_error_deg = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<f64>()
                        .map_err(|e| format!("invalid --loop-closure-max-rot-error: {e}"))?;
                }
                "--local-ba-window" => {
                    local_ba_window = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --local-ba-window: {error}"))?;
                }
                "--local-ba-iterations" => {
                    local_ba_iterations = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --local-ba-iterations: {error}"))?;
                }
                "--no-loop-matching" => no_loop_matching = true,
                "--loop-match-max-rot-error" => {
                    loop_match_max_rotation_error_deg = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<f64>()
                            .map_err(|e| format!("invalid --loop-match-max-rot-error: {e}"))?,
                    )
                }
                "--loop-closure-factors" => loop_closure_factors = true,
                "--loop-closure-min-corr" => {
                    loop_closure_min_correspondences = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --loop-closure-min-corr: {error}"))?;
                }
                "--loop-closure-weight" => {
                    loop_closure_weight = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<f64>()
                        .map_err(|error| format!("invalid --loop-closure-weight: {error}"))?;
                }
                "--match-top-k" => {
                    match_top_k = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<usize>()
                            .map_err(|error| format!("invalid --match-top-k: {error}"))?,
                    );
                }
                "--imu-preintegration-weight" => {
                    imu_preintegration_weight = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<f64>()
                            .map_err(|error| {
                                format!("invalid --imu-preintegration-weight: {error}")
                            })?,
                    );
                }
                "--joint-vi-ba-weight" => {
                    joint_vi_ba_weight = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<f64>()
                            .map_err(|error| format!("invalid --joint-vi-ba-weight: {error}"))?,
                    );
                }
                "--relative-pose-weight" => {
                    relative_pose_weight = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<f64>()
                            .map_err(|error| format!("invalid --relative-pose-weight: {error}"))?,
                    );
                }
                "--roll-pitch-weight" => {
                    roll_pitch_weight = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<f64>()
                            .map_err(|error| format!("invalid --roll-pitch-weight: {error}"))?,
                    );
                }
                "--projection-host-window" => {
                    projection_host_window = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<u64>()
                            .map_err(|error| {
                                format!("invalid --projection-host-window: {error}")
                            })?,
                    );
                }
                "--projection-radius" => {
                    projection_radius_px = Some(
                        next(&mut arguments, &option)?
                            .to_string_lossy()
                            .parse::<f64>()
                            .map_err(|error| format!("invalid --projection-radius: {error}"))?,
                    );
                }
                "--num-opt-iter" => {
                    num_opt_iter = next(&mut arguments, &option)?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --num-opt-iter: {error}"))?;
                    if num_opt_iter == 0 {
                        return Err("--num-opt-iter must be positive".into());
                    }
                }
                unknown => return Err(format!("unknown option `{unknown}`\n\n{}", Self::usage())),
            }
        }
        Ok(Self {
            euroc_dir: euroc_dir
                .ok_or_else(|| format!("--euroc-dir is required\n\n{}", Self::usage()))?,
            calibration: calibration
                .ok_or_else(|| format!("--calibration is required\n\n{}", Self::usage()))?,
            config,
            out_dir,
            frontend_stats_csv,
            max_frames,
            optimize_every_k,
            periodic_iterations,
            realtime,
            pipeline,
            pipeline_explicit,
            pipeline_capacity,
            decode_threads,
            threads,
            mapper_queue_capacity,
            no_urgent_keyframes,
            imu_seed_klt,
            retained_marg_diagnostics,
            num_opt_iter,
            match_top_k,
            projection_rematch,
            incremental_local_mapping,
            projection_host_window,
            projection_radius_px,
            no_loop_matching,
            loop_match_max_rotation_error_deg,
            loop_closure_factors,
            loop_closure_min_correspondences,
            loop_closure_weight,
            loop_closure_max_rotation_error_deg,
            local_ba_window,
            local_ba_iterations,
            imu_preintegration_weight,
            joint_vi_ba_weight,
            relative_pose_weight,
            roll_pitch_weight,
        })
    }

    fn usage() -> String {
        "usage: basalt_euroc_online_slam_demo --euroc-dir DIR --calibration FILE \
         [--config FILE] [--out-dir DIR] [--max-frames N] [--optimize-every-k K] \
         [--periodic-iterations N] [--realtime | --as-fast-as-possible] \
         [--pipeline | --no-pipeline] [--pipeline-capacity N] [--decode-threads N] [--threads N] \
         [--mapper-queue-capacity N] [--retained-marg-diagnostics] [--num-opt-iter N] \
         [--match-top-k N] [--frontend-stats-csv <path>] \
         [--projection-rematch] [--local-mapping] [--imu-seed-klt] \
         [--projection-host-window N] [--projection-radius PX] \
         [--loop-closure-factors] [--loop-closure-min-corr N] [--loop-closure-weight W] \
         [--loop-closure-max-rot-error DEG] [--local-ba-window N] [--local-ba-iterations N] \
         [--relative-pose-weight W] [--roll-pitch-weight W] [--imu-preintegration-weight W] \
         [--joint-vi-ba-weight W]"
            .into()
    }
}

fn next<I>(arguments: &mut I, option: &str) -> Result<std::ffi::OsString, String>
where
    I: Iterator<Item = std::ffi::OsString>,
{
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_requires_euroc_dir_and_calibration() {
        assert!(Args::parse(["--calibration", "c.json"].map(Into::into)).is_err());
        assert!(Args::parse(["--euroc-dir", "d"].map(Into::into)).is_err());
    }

    #[test]
    fn parser_defaults_to_as_fast_as_possible() {
        let args = Args::parse(["--euroc-dir", "d", "--calibration", "c.json"].map(Into::into))
            .expect("parses");
        assert!(!args.realtime);
        assert_eq!(args.frontend_stats_csv, None);
        assert_eq!(
            args.optimize_every_k,
            OnlineMapperConfig::default().optimize_every_k
        );
        assert_eq!(
            args.periodic_iterations,
            OnlineMapperConfig::default().periodic_iterations
        );
        assert_eq!(args.imu_preintegration_weight, None);
        assert_eq!(args.joint_vi_ba_weight, None);
        assert_eq!(args.relative_pose_weight, None);
        assert_eq!(args.roll_pitch_weight, None);
    }

    #[test]
    fn parser_accepts_imu_preintegration_weight() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--imu-preintegration-weight",
                "3.5",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert_eq!(args.imu_preintegration_weight, Some(3.5));
    }

    #[test]
    fn parser_accepts_joint_vi_ba_weight() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--joint-vi-ba-weight",
                "2.5",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert_eq!(args.joint_vi_ba_weight, Some(2.5));
    }

    #[test]
    fn parser_accepts_relative_pose_and_roll_pitch_weight() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--relative-pose-weight",
                "4",
                "--roll-pitch-weight",
                "2.5",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert_eq!(args.relative_pose_weight, Some(4.0));
        assert_eq!(args.roll_pitch_weight, Some(2.5));
    }

    #[test]
    fn imu_seed_klt_defaults_off_and_can_be_enabled() {
        let base = ["--euroc-dir", "d", "--calibration", "c.json"];
        let args = Args::parse(base.map(Into::into)).expect("parses");
        assert!(!args.imu_seed_klt);
        let args = Args::parse(base.into_iter().chain(["--imu-seed-klt"]).map(Into::into))
            .expect("parses");
        assert!(args.imu_seed_klt);
    }

    #[test]
    fn parser_accepts_frontend_stats_csv() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--frontend-stats-csv",
                "foo.csv",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert_eq!(args.frontend_stats_csv, Some(PathBuf::from("foo.csv")));
    }

    #[test]
    fn parser_accepts_realtime_and_k() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--realtime",
                "--optimize-every-k",
                "5",
                "--periodic-iterations",
                "2",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert!(args.realtime);
        assert_eq!(args.optimize_every_k, 5);
        assert_eq!(args.periodic_iterations, 2);
    }

    #[test]
    fn parser_defaults_pipeline_on_bit_identical_to_serial() {
        // `--pipeline` defaults on: it is bit-identical to the serial path
        // (verified separately: identical trajectory.tum SHA-256 and
        // identical marg_data/ contents on MH_03/MH_04) and materially
        // faster, so it is the canonical default rather than an opt-in.
        let args = Args::parse(["--euroc-dir", "d", "--calibration", "c.json"].map(Into::into))
            .expect("parses");
        assert!(args.pipeline);
        assert!(!args.pipeline_explicit);
        assert_eq!(args.pipeline_capacity, 4);
        assert_eq!(args.decode_threads, 3);
        assert_eq!(args.threads, None);
        assert_eq!(args.mapper_queue_capacity, DEFAULT_MAPPER_QUEUE_CAPACITY);
    }

    #[test]
    fn parser_accepts_no_pipeline_to_restore_serial_default() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--no-pipeline",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert!(!args.pipeline);
        assert!(args.pipeline_explicit);
    }

    #[test]
    fn realtime_silently_falls_back_to_serial_when_pipeline_is_only_the_default() {
        assert_eq!(resolve_pipeline_realtime(true, false, true), Ok(false));
    }

    #[test]
    fn realtime_rejects_explicit_pipeline_request() {
        assert_eq!(
            resolve_pipeline_realtime(true, true, true),
            Err("--realtime is not supported together with --pipeline")
        );
    }

    #[test]
    fn realtime_leaves_pipeline_alone_when_not_paced() {
        assert_eq!(resolve_pipeline_realtime(true, false, false), Ok(true));
        assert_eq!(resolve_pipeline_realtime(false, true, true), Ok(false));
    }

    #[test]
    fn urgent_keyframes_default_on_and_can_be_disabled() {
        let base = ["--euroc-dir", "d", "--calibration", "c.json"];
        let args = Args::parse(base.map(Into::into)).expect("parses");
        assert!(!args.no_urgent_keyframes);
        let args = Args::parse(
            base.into_iter()
                .chain(["--no-urgent-keyframes"])
                .map(Into::into),
        )
        .expect("parses");
        assert!(args.no_urgent_keyframes);
    }

    #[test]
    fn parser_accepts_mapper_queue_capacity_override() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--mapper-queue-capacity",
                "16",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert_eq!(args.mapper_queue_capacity, 16);
    }

    #[test]
    fn parser_rejects_zero_mapper_queue_capacity() {
        assert!(Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--mapper-queue-capacity",
                "0"
            ]
            .map(Into::into)
        )
        .is_err());
    }

    #[test]
    fn parser_accepts_pipeline_and_thread_flags() {
        let args = Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--pipeline",
                "--pipeline-capacity",
                "8",
                "--decode-threads",
                "2",
                "--threads",
                "4",
            ]
            .map(Into::into),
        )
        .expect("parses");
        assert!(args.pipeline);
        assert_eq!(args.pipeline_capacity, 8);
        assert_eq!(args.decode_threads, 2);
        assert_eq!(args.threads, Some(4));
    }

    #[test]
    fn parser_rejects_zero_pipeline_capacity_and_threads() {
        assert!(Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--pipeline-capacity",
                "0"
            ]
            .map(Into::into)
        )
        .is_err());
        assert!(Args::parse(
            [
                "--euroc-dir",
                "d",
                "--calibration",
                "c.json",
                "--threads",
                "0"
            ]
            .map(Into::into)
        )
        .is_err());
    }

    fn se3(tx: f64, ty: f64, tz: f64, yaw_deg: f64) -> SE3 {
        SE3::new(
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, yaw_deg.to_radians()),
            nalgebra::Vector3::new(tx, ty, tz),
        )
    }

    #[test]
    fn propagation_is_exact_at_keyframes() {
        let mut vio = BTreeMap::new();
        vio.insert(0, (0, se3(0.0, 0.0, 0.0, 0.0)));
        vio.insert(10, (10, se3(1.0, 0.0, 0.0, 0.0)));
        let mut mapper = BTreeMap::new();
        // A keyframe correction that moves frame 10 sideways and rotates it.
        mapper.insert(0, se3(0.0, 0.0, 0.0, 0.0));
        mapper.insert(10, se3(1.5, 0.2, 0.0, 5.0));

        let propagated = propagate_to_all_frames(&vio, &mapper).expect("propagates");
        let by_frame: BTreeMap<i64, &SE3> = propagated
            .iter()
            .map(|(timestamp_ns, pose)| (*timestamp_ns, pose))
            .collect();
        let corrected_10 = by_frame[&10];
        assert!((corrected_10.translation - mapper[&10].translation).norm() < 1e-9);
        assert!(corrected_10.rotation.angle_to(&mapper[&10].rotation).abs() < 1e-9);
    }

    #[test]
    fn interpolated_propagation_is_exact_at_keyframes_and_blends_between() {
        let mut vio = BTreeMap::new();
        for frame in 0..=20_u64 {
            vio.insert(
                frame,
                (frame as i64 * 10, se3(frame as f64 * 0.1, 0.0, 0.0, 0.0)),
            );
        }
        let mut mapper = BTreeMap::new();
        // Keyframe 5 is corrected by +0.2 m in y, keyframe 15 by +0.6 m in y.
        mapper.insert(5, se3(0.5, 0.2, 0.0, 0.0));
        mapper.insert(15, se3(1.5, 0.6, 0.0, 0.0));
        let interpolated = propagate_interpolated(&vio, &mapper).expect("propagates");
        let by_time: BTreeMap<i64, &SE3> = interpolated.iter().map(|(t, p)| (*t, p)).collect();
        // Exact at keyframes.
        assert!((by_time[&50].translation - mapper[&5].translation).norm() < 1e-12);
        assert!((by_time[&150].translation - mapper[&15].translation).norm() < 1e-12);
        // Halfway between the keyframes the y correction is halfway too.
        assert!((by_time[&100].translation.y - 0.4).abs() < 1e-12);
        assert!((by_time[&100].translation.x - 1.0).abs() < 1e-12);
        // Before the first / after the last keyframe: that keyframe's delta.
        assert!((by_time[&0].translation.y - 0.2).abs() < 1e-12);
        assert!((by_time[&200].translation.y - 0.6).abs() < 1e-12);
    }

    #[test]
    fn propagation_keeps_relative_motion_to_nearest_preceding_keyframe() {
        let mut vio = BTreeMap::new();
        vio.insert(0, (0, se3(0.0, 0.0, 0.0, 0.0)));
        vio.insert(5, (5, se3(0.5, 0.0, 0.0, 0.0))); // non-keyframe, between 0 and 10
        vio.insert(10, (10, se3(1.0, 0.0, 0.0, 0.0)));
        let mut mapper = BTreeMap::new();
        mapper.insert(0, se3(2.0, 0.0, 0.0, 0.0)); // shift everything by +2.0 in x
        mapper.insert(10, se3(3.0, 0.0, 0.0, 0.0));

        let propagated = propagate_to_all_frames(&vio, &mapper).expect("propagates");
        let by_frame: BTreeMap<i64, &SE3> = propagated
            .iter()
            .map(|(timestamp_ns, pose)| (*timestamp_ns, pose))
            .collect();
        // Frame 5's raw VIO relative motion from keyframe 0 is +0.5 in x;
        // keyframe 0's correction shifts everything by +2.0, so frame 5
        // should land at 2.5, not be independently corrected.
        assert!((by_frame[&5].translation.x - 2.5).abs() < 1e-9);
    }

    #[test]
    fn propagation_rejects_a_keyframe_missing_from_the_vio_trajectory() {
        let mut vio = BTreeMap::new();
        vio.insert(0, (0, se3(0.0, 0.0, 0.0, 0.0)));
        let mut mapper = BTreeMap::new();
        mapper.insert(0, se3(0.0, 0.0, 0.0, 0.0));
        mapper.insert(99, se3(1.0, 0.0, 0.0, 0.0)); // not in vio_trajectory
        assert!(propagate_to_all_frames(&vio, &mapper).is_err());
    }
}
