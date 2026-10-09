//! Sensor-only EuRoC replay through the dedicated Basalt frontend/estimator.
//!
//! The command reads cam0/cam1 PNGs, IMU CSV, the upstream Double Sphere
//! calibration JSON, and the upstream `euroc_config.json`.  It deliberately
//! has no argument or code path for reference trajectories: all outputs are
//! produced causally from sensor data and estimator state.
//!
//! ```text
//! cargo run --release --example basalt_euroc_vio_demo -- \
//!   --euroc-dir /data/MH_01_easy \
//!   --calibration /data/euroc_ds_calib.json \
//!   --config configs/basalt/euroc_config.json \
//!   --out-dir target/basalt_mh01_smoke --max-frames 80
//! ```
//!
//! `--mono` replays the same recording as monocular-inertial VIO (cam0 +
//! IMU): `mav0/cam1` is not read and the (stereo) calibration is reduced to
//! camera 0, so every landmark is seeded by temporal triangulation and metric
//! scale comes from the IMU. It is opt-in and experimental; see
//! `docs/mono_inertial_vio.md`. Without it the stereo replay is unchanged.

use std::{
    env, fs,
    io::{BufWriter, Write},
    path::PathBuf,
    process,
};

use visloc_basalt::{
    vio::{
        AomBlockData, ImuLinkDiagnostics, LmRunDiagnostics, NativeCompanionIdentity,
        WindowDiagnostics,
    },
    BasaltAdapterError, BasaltAdapterOutput, BasaltNavState, BasaltVioEstimatorAdapter,
    EurocSensorDataset, TimingBreakdown, TimingBucket,
};

#[derive(Debug)]
struct Args {
    euroc_dir: PathBuf,
    calibration: PathBuf,
    config: PathBuf,
    out_dir: PathBuf,
    max_frames: Option<usize>,
    no_trace: bool,
    no_marg_data: bool,
    retained_marg_diagnostics: bool,
    native_companion_binding: Option<PathBuf>,
    pipeline: bool,
    pipeline_capacity: usize,
    decode_threads: usize,
    threads: Option<usize>,
    mono: bool,
    /// `--dump-tracks FILE`: write every frame's tracked keypoints as CSV
    /// (`frame_id,timestamp_ns,camera_id,track_id,x,y`), e.g. for rendering
    /// tracking overlays. Off by default.
    dump_tracks: Option<PathBuf>,
}

const NATIVE_COMPANION_BINDING_SCHEMA: &str = "basalt.rust.native_companion_binding.v1";

fn main() {
    if let Err(error) = run() {
        eprintln!("basalt_euroc_vio_demo: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args_os().skip(1))
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    if args.no_marg_data && args.native_companion_binding.is_some() {
        return Err("--native-companion-binding requires MargData output".into());
    }
    if args.mono && args.native_companion_binding.is_some() {
        return Err(
            "--native-companion-binding binds stereo native captures; not valid with --mono".into(),
        );
    }
    // Sizes the process-wide rayon pool used by data-parallel stages (e.g.
    // the frontend's per-track temporal KLT). Left unset, rayon lazily sizes
    // its default global pool to `std::thread::available_parallelism()` on
    // first use -- no numeric behavior depends on this count, only wall time.
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("failed to configure rayon thread pool: {error}"))?;
    }
    // Load the immutable native identity before replay or output creation.
    // This makes the resulting packet a capture-time paired artifact rather
    // than a JSON document labelled after the estimator has finished.
    let native_companion_identity = args
        .native_companion_binding
        .as_ref()
        .map(|path| load_native_companion_identity(path))
        .transpose()?;
    fs::create_dir_all(&args.out_dir)?;
    let marg_dir = if args.no_marg_data {
        None
    } else {
        let path = args.out_dir.join("marg_data");
        fs::create_dir_all(&path)?;
        Some(path)
    };
    // Trace lines are streamed straight to disk as they are produced rather
    // than accumulated in one in-process String for the whole replay.  A
    // long, wide-FOV sequence (e.g. LaMAria's ~18.3k-frame sequence_1_19)
    // emits a rich per-frame window/LM/AOM diagnostic line; buffering all of
    // them for the run's lifetime makes the buffer's capacity-doubling
    // reallocation the largest live allocation in the process, and on a
    // long enough sequence that doubling eventually asks the allocator for
    // more contiguous memory than is available (observed: an 18,723,373,056
    // byte request at frame 10849 of sequence_1_19). Streaming keeps peak
    // trace memory at one line, independent of sequence length, and is
    // byte-identical to the old buffered contents once concatenated.
    let trace_path = args.out_dir.join("trace.jsonl");
    let mut trace_writer = if args.no_trace {
        None
    } else {
        Some(BufWriter::new(fs::File::create(&trace_path)?))
    };
    let mut tracks_writer = match args.dump_tracks.as_ref() {
        Some(path) => {
            let mut writer = BufWriter::new(fs::File::create(path)?);
            writer.write_all(b"frame_id,timestamp_ns,camera_id,track_id,x,y\n")?;
            Some(writer)
        }
        None => None,
    };

    let mut timing = TimingBreakdown::from_env();
    let dataset = match (args.mono, timing.enabled()) {
        (false, true) => EurocSensorDataset::open_with_timing(
            &args.euroc_dir,
            &args.calibration,
            &args.config,
            &mut timing,
        )?,
        (false, false) => {
            EurocSensorDataset::open(&args.euroc_dir, &args.calibration, &args.config)?
        }
        (true, true) => EurocSensorDataset::open_monocular_with_timing(
            &args.euroc_dir,
            &args.calibration,
            &args.config,
            &mut timing,
        )?,
        (true, false) => {
            EurocSensorDataset::open_monocular(&args.euroc_dir, &args.calibration, &args.config)?
        }
    };
    let mut adapter =
        BasaltVioEstimatorAdapter::from_config(dataset.calibration(), dataset.config())?;
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

    let mut tum = String::new();
    let mut csv = String::from(
        "frame_id,timestamp_ns,tx,ty,tz,qw,qx,qy,qz,cam0_observations,cam1_observations,imu_samples\n",
    );
    let mut total_imu = 0usize;
    let mut total_observations = 0usize;
    let mut last_timestamp_ns = None;
    let mut mapper_packet_ordinal = 0u64;
    let mut native_companion_bound = false;
    let mut demo_index = 0usize;

    let mode = adapter_process_mode(args.no_trace, args.no_marg_data);
    let (retain_marg_data, retain_trace) = match mode {
        AdapterProcessMode::Full => (true, true),
        AdapterProcessMode::WithoutMargData => (false, true),
        AdapterProcessMode::WithoutMargDataAndTrace => (false, false),
    };

    // Shared per-frame trajectory/trace/MargData writer for both the serial
    // loop below and the two-thread `--pipeline` path.  Running the exact
    // same body from both call sites -- only the wall-clock overlap between
    // frontend and estimator differs -- is what keeps their outputs
    // identical: this closure never touches the estimator's or frontend's
    // arithmetic, only serializes what they already produced.
    let mut handle_output = |output: BasaltAdapterOutput,
                             demo_timing: &mut TimingBreakdown|
     -> Result<(), BasaltAdapterError> {
        let timestamp_ns = output.tracks.frame.timestamp_ns;
        if last_timestamp_ns.is_some_and(|previous| timestamp_ns <= previous) {
            return Err(BasaltAdapterError::Output(format!(
                "non-monotonic output timestamp at frame {demo_index}"
            )));
        }
        last_timestamp_ns = Some(timestamp_ns);
        total_imu += output.imu_count;
        total_observations += output.tracks.observations.len();

        demo_timing
            .measure(
                TimingBucket::DemoOutput,
                || -> Result<(), Box<dyn std::error::Error>> {
                    append_trajectory(&mut tum, &mut csv, &output);
                    if let Some(writer) = trace_writer.as_mut() {
                        writer.write_all(trace_json(&output).as_bytes())?;
                    }
                    if let Some(writer) = tracks_writer.as_mut() {
                        for observation in &output.tracks.observations {
                            writeln!(
                                writer,
                                "{},{},{},{},{:.2},{:.2}",
                                observation.frame_id,
                                observation.timestamp_ns,
                                observation.camera_id,
                                observation.track_id,
                                observation.pixel.x,
                                observation.pixel.y
                            )?;
                        }
                    }
                    // Basalt computes state-only marginalization on many frames, but its
                    // mapper queue receives MargData only for a selected KF removal.
                    // Keep the diagnostic record in the in-process output while writing
                    // only actual mapper packets to the on-disk stream.
                    if let Some(marg_dir) = marg_dir.as_ref() {
                        if output.estimator.marg_data.is_mapper_packet() {
                            let bind_identity = native_companion_identity
                                .as_ref()
                                .filter(|identity| identity.event_ordinal == mapper_packet_ordinal);
                            if let Some(identity) = bind_identity {
                                output
                                    .estimator
                                    .marg_data
                                    .validate_native_companion_identity(
                                        identity,
                                        mapper_packet_ordinal,
                                    )?;
                            }
                            let marg_path = marg_dir
                                .join(format!("frame_{:06}.json", output.tracks.frame.frame_id));
                            let file = fs::File::create(marg_path)?;
                            let mut writer = BufWriter::new(file);
                            if let Some(identity) = bind_identity {
                                output
                                    .estimator
                                    .marg_data
                                    .write_mapper_packet_json_with_native_companion_identity(
                                        &mut writer,
                                        identity,
                                        mapper_packet_ordinal,
                                    )?;
                                native_companion_bound = true;
                            } else {
                                output
                                    .estimator
                                    .marg_data
                                    .write_mapper_packet_json(&mut writer)?;
                            }
                            writer.flush()?;
                            mapper_packet_ordinal += 1;
                        }
                    }
                    Ok(())
                },
            )
            .map_err(|error| BasaltAdapterError::Output(error.to_string()))?;

        if demo_index == 0 || (demo_index + 1).is_multiple_of(10) || demo_index + 1 == frame_limit {
            eprintln!(
                "frame={} timestamp_ns={} observations={} imu={} created={} retained={} rejected={}",
                output.tracks.frame.frame_id,
                timestamp_ns,
                output.tracks.observations.len(),
                output.imu_count,
                output.tracks.created_track_ids.len(),
                output.tracks.retained_track_ids.len(),
                output.tracks.rejected_track_ids.len(),
            );
        }
        demo_index += 1;
        Ok(())
    };

    if args.pipeline {
        // Two-thread pipeline: a frontend/producer thread (dataset
        // acquisition + `DirectKltStream` tracking) overlapped in wall time
        // with the estimator/consumer thread running on this thread, mirroring
        // upstream Basalt's `OpticalFlow` thread ‖ estimator thread split.
        // Frame order and every per-frame computation are unchanged from the
        // serial path below; see `BasaltVioEstimatorAdapter::process_euroc_stream_pipelined`.
        let (producer_timing, consumer_timing) = adapter.process_euroc_stream_pipelined(
            &dataset,
            frame_limit,
            retain_marg_data,
            retain_trace,
            args.pipeline_capacity,
            args.decode_threads,
            handle_output,
        )?;
        timing.merge_from(&producer_timing);
        timing.merge_from(&consumer_timing);
    } else {
        for index in 0..frame_limit {
            let sensor_frame = if timing.enabled() {
                timing.measure_with(TimingBucket::DatasetFrameAcquisition, |timing| {
                    dataset.frame_with_timing(index, timing)
                })?
            } else {
                dataset.frame(index)?
            };
            let output = match mode {
                AdapterProcessMode::Full => adapter.process(sensor_frame)?,
                AdapterProcessMode::WithoutMargData => {
                    adapter.process_without_marg_data(sensor_frame)?
                }
                AdapterProcessMode::WithoutMargDataAndTrace => {
                    adapter.process_without_marg_data_no_trace(sensor_frame)?
                }
            };
            handle_output(output, &mut timing)?;
        }
    }

    if let Some(identity) = native_companion_identity.as_ref() {
        if !native_companion_bound {
            return Err(format!(
                "native companion event ordinal {} was not emitted in this replay",
                identity.event_ordinal
            )
            .into());
        }
    }

    let trajectory_tum = args.out_dir.join("trajectory.tum");
    let trajectory_csv = args.out_dir.join("trajectory.csv");
    let trace_summary = if args.no_trace {
        "disabled".to_owned()
    } else {
        trace_path.display().to_string()
    };
    let marg_summary = marg_dir
        .as_ref()
        .map_or_else(|| "disabled".to_owned(), |path| path.display().to_string());

    let mut summary = format!(
        "sensor_only=true\nframes_requested={}\nframes_processed={}\ncam0_manifest_frames={}\ncam1_manifest_timestamps={}\nimu_samples_loaded={}\nimu_samples_delivered={}\nobservations_emitted={}\ntrajectory_tum={}\ntrajectory_csv={}\ntrace_jsonl={}\nmarg_data_dir={}\n",
        args.max_frames
            .map_or_else(|| "all".to_owned(), |value| value.to_string()),
        frame_limit,
        dataset.cam0_manifest_count(),
        dataset.cam1_timestamp_count(),
        dataset.imu_samples().len(),
        total_imu,
        total_observations,
        trajectory_tum.display(),
        trajectory_csv.display(),
        trace_summary,
        marg_summary,
    );
    // Only the opt-in monocular replay adds a line, so the stereo summary
    // stays byte-identical for existing tooling.
    if dataset.is_monocular() {
        summary.push_str("camera_mode=mono_inertial\n");
    }

    timing.measure(TimingBucket::DemoTrajectoryOutput, || {
        fs::write(&trajectory_tum, tum)?;
        fs::write(&trajectory_csv, csv)?;
        if let Some(mut writer) = trace_writer {
            writer.flush()?;
        }
        if let Some(mut writer) = tracks_writer {
            writer.flush()?;
        }
        Ok::<(), std::io::Error>(())
    })?;

    let timing_path = args.out_dir.join("timing_breakdown.json");
    timing.measure_with(TimingBucket::DemoTeardown, |timing| {
        if timing.enabled() {
            // The `--pipeline` path already merged the frontend/estimator
            // producer and consumer collectors (including the estimator's
            // internal LM/Estimator* buckets) into `timing` above; merging
            // `adapter.timing_breakdown_with_estimator()` again here would
            // double-count those buckets. The serial path never touches the
            // adapter's own `self.timing` field or reads the estimator's
            // internal collector until this point, so it still needs the
            // merge.
            if !args.pipeline {
                let adapter_timing = adapter.timing_breakdown_with_estimator();
                timing.merge_from(&adapter_timing);
            }
            // All trajectory/output strings are already materialized.  The
            // collector is merged before the explicit drops, making the
            // process-side teardown gap visible without changing the normal
            // (timing-disabled) drop order or any solver/output arithmetic.
            drop(adapter);
            drop(dataset);
        }
        fs::write(args.out_dir.join("summary.txt"), &summary)?;
        Ok::<(), std::io::Error>(())
    })?;
    if timing.enabled() {
        timing.write_json(&timing_path)?;
    }
    println!("{summary}");
    Ok(())
}

fn load_native_companion_identity(
    path: &std::path::Path,
) -> Result<NativeCompanionIdentity, Box<dyn std::error::Error>> {
    let binding: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let object = binding
        .as_object()
        .ok_or("native companion binding must be a JSON object")?;
    if object.len() != 2
        || !object.contains_key("schema")
        || !object.contains_key("native_companion_identity")
    {
        return Err("native companion binding has missing or unknown top-level fields".into());
    }
    let schema = object["schema"]
        .as_str()
        .ok_or("native companion binding schema must be a string")?;
    if schema != NATIVE_COMPANION_BINDING_SCHEMA {
        return Err(format!("unsupported native companion binding schema {schema:?}").into());
    }
    serde_json::from_value(object["native_companion_identity"].clone())
        .map_err(|error| error.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdapterProcessMode {
    Full,
    WithoutMargData,
    WithoutMargDataAndTrace,
}

/// Select the adapter's retention mode from the two independent CLI flags.
///
/// A trace-only suppression cannot use the lean estimator path because that
/// path also suppresses MargData.  In that quadrant the adapter keeps the
/// existing full processing contract while the demo omits the trace file.
const fn adapter_process_mode(no_trace: bool, no_marg_data: bool) -> AdapterProcessMode {
    match (no_marg_data, no_trace) {
        (false, _) => AdapterProcessMode::Full,
        (true, false) => AdapterProcessMode::WithoutMargData,
        (true, true) => AdapterProcessMode::WithoutMargDataAndTrace,
    }
}

fn append_trajectory(tum: &mut String, csv: &mut String, output: &BasaltAdapterOutput) {
    let state = &output.estimator.state;
    let pose = &state.imu_to_world;
    let q = pose.rotation.quaternion();
    let timestamp_seconds = output.tracks.frame.timestamp_ns as f64 * 1.0e-9;
    let cam0_count = output
        .tracks
        .observations
        .iter()
        .filter(|observation| observation.camera_id == 0)
        .count();
    let cam1_count = output
        .tracks
        .observations
        .iter()
        .filter(|observation| observation.camera_id == 1)
        .count();
    tum.push_str(&format!(
        "{timestamp_seconds:.9} {:.9} {:.9} {:.9} {:.12} {:.12} {:.12} {:.12}\n",
        pose.translation.x, pose.translation.y, pose.translation.z, q.i, q.j, q.k, q.w,
    ));
    csv.push_str(&format!(
        "{},{},{:.9},{:.9},{:.9},{:.12},{:.12},{:.12},{:.12},{cam0_count},{cam1_count},{}\n",
        output.tracks.frame.frame_id,
        output.tracks.frame.timestamp_ns,
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        q.w,
        q.i,
        q.j,
        q.k,
        output.imu_count,
    ));
}

fn trace_json(output: &BasaltAdapterOutput) -> String {
    let tracks = &output.tracks;
    let state = &output.estimator.state;
    let pose = &state.imu_to_world;
    let q = pose.rotation.quaternion();
    let stages = tracks
        .stage_trace
        .iter()
        .map(|stage| format!("\"{stage:?}\""))
        .collect::<Vec<_>>()
        .join(",");
    let phases = output
        .estimator
        .phases
        .iter()
        .map(|phase| format!("\"{phase:?}\""))
        .collect::<Vec<_>>()
        .join(",");
    let rejects = tracks
        .reject_counters
        .iter()
        .map(|(reason, count)| format!("{{\"reason\":\"{reason:?}\",\"count\":{count}}}"))
        .collect::<Vec<_>>()
        .join(",");
    let window = window_json(&output.estimator.window);
    let state_trace = state_trace_json(&output.estimator.state_trace);
    let marg = &output.estimator.marg_data;
    let targets = &marg.marginalization;
    format!(
        "{{\"frame_id\":{},\"timestamp_ns\":{},\"imu_count\":{},\"observation_count\":{},\"created_track_ids\":{},\"retained_track_ids\":{},\"rejected_track_ids\":{},\"stage_trace\":[{}],\"vio_phases\":[{}],\"rejects\":[{}],\"is_keyframe\":{},\"connected_cam0\":{},\"unconnected_cam0\":{},\"active_state_count\":{},\"active_pose_count\":{},\"aom_order\":{},\"kfs_to_marg\":{},\"marginalization\":{{\"poses_to_marg\":{},\"states_to_marg_all\":{},\"states_to_marg_vel_bias\":{},\"lost_landmarks\":{}}},\"state\":{{\"tx\":{:.17e},\"ty\":{:.17e},\"tz\":{:.17e},\"qw\":{:.17e},\"qx\":{:.17e},\"qy\":{:.17e},\"qz\":{:.17e},\"velocity\":[{:.17e},{:.17e},{:.17e}],\"gyro_bias\":[{:.17e},{:.17e},{:.17e}],\"accel_bias\":[{:.17e},{:.17e},{:.17e}]}} ,\"state_trace\":{},\"window_attempted\":{},\"imu_integration_fallback\":{},\"window\":{},\"marg_data_hash\":{}}}\n",
        tracks.frame.frame_id,
        tracks.frame.timestamp_ns,
        output.imu_count,
        tracks.observations.len(),
        json_u64_array(&tracks.created_track_ids),
        json_u64_array(&tracks.retained_track_ids),
        json_u64_array(&tracks.rejected_track_ids),
        stages,
        phases,
        rejects,
        output.estimator.is_keyframe,
        output.estimator.connected_cam0,
        output.estimator.unconnected_cam0,
        output.estimator.active_state_count,
        output.estimator.active_pose_count,
        aom_order_json(&marg.aom_order),
        json_u64_array(&marg.kfs_to_marg),
        json_u64_array(&targets.poses_to_marg),
        json_u64_array(&targets.states_to_marg_all),
        json_u64_array(&targets.states_to_marg_vel_bias),
        json_u64_array(&targets.lost_landmarks),
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        q.w,
        q.i,
        q.j,
        q.k,
        state.velocity_world_m_s.x,
        state.velocity_world_m_s.y,
        state.velocity_world_m_s.z,
        state.gyro_bias_rad_s.x,
        state.gyro_bias_rad_s.y,
        state.gyro_bias_rad_s.z,
        state.accel_bias_m_s2.x,
        state.accel_bias_m_s2.y,
        state.accel_bias_m_s2.z,
        state_trace,
        output.estimator.window.attempted,
        output.estimator.imu_integration_fallback,
        window,
        output.estimator.marg_data.stable_hash(),
    )
}

fn state_trace_json(trace: &visloc_basalt::vio::EstimatorStateTrace) -> String {
    format!(
        "{{\"state_from\":{},\"initialization_output\":{},\"predicted_state\":{},\"post_opt_state\":{},\"imu_propagation\":{},\"imu_integration_fallback\":{}}}",
        trace
            .state_from
            .as_ref().map_or_else(|| "null".into(), nav_state_json),
        trace
            .initialization_output
            .as_ref().map_or_else(|| "null".into(), nav_state_json),
        nav_state_json(&trace.predicted_state),
        nav_state_json(&trace.post_opt_state),
        trace
            .imu_propagation
            .as_ref().map_or_else(|| "null".into(), imu_propagation_json),
        trace.imu_integration_fallback,
    )
}

fn imu_propagation_json(trace: &visloc_basalt::vio::ImuPropagationTrace) -> String {
    let q = trace.delta_rotation.quaternion();
    format!(
        "{{\"interval_start_ns\":{},\"interval_end_ns\":{},\"sample_timestamps_ns\":{},\"delta_time\":{},\"delta_position\":{},\"delta_rotation_xyzw\":[{},{},{},{}],\"delta_velocity\":{}}}",
        trace.interval_start_ns,
        trace.interval_end_ns,
        json_array_i64(&trace.sample_timestamps_ns),
        json_number(trace.delta_time),
        json_vec3([
            trace.delta_position.x,
            trace.delta_position.y,
            trace.delta_position.z,
        ]),
        json_number(q.i),
        json_number(q.j),
        json_number(q.k),
        json_number(q.w),
        json_vec3([
            trace.delta_velocity.x,
            trace.delta_velocity.y,
            trace.delta_velocity.z,
        ]),
    )
}

fn nav_state_json(state: &BasaltNavState) -> String {
    let pose = &state.imu_to_world;
    let q = pose.rotation.quaternion();
    format!(
        "{{\"tx\":{},\"ty\":{},\"tz\":{},\"qw\":{},\"qx\":{},\"qy\":{},\"qz\":{},\"velocity\":{},\"gyro_bias\":{},\"accel_bias\":{}}}",
        json_number(pose.translation.x),
        json_number(pose.translation.y),
        json_number(pose.translation.z),
        json_number(q.w),
        json_number(q.i),
        json_number(q.j),
        json_number(q.k),
        json_vec3([
            state.velocity_world_m_s.x,
            state.velocity_world_m_s.y,
            state.velocity_world_m_s.z,
        ]),
        json_vec3([
            state.gyro_bias_rad_s.x,
            state.gyro_bias_rad_s.y,
            state.gyro_bias_rad_s.z,
        ]),
        json_vec3([
            state.accel_bias_m_s2.x,
            state.accel_bias_m_s2.y,
            state.accel_bias_m_s2.z,
        ]),
    )
}

fn window_json(diagnostics: &WindowDiagnostics) -> String {
    let lm = diagnostics
        .lm
        .iter()
        .map(lm_run_json)
        .collect::<Vec<_>>()
        .join(",");
    let imu_links = diagnostics
        .imu_links
        .iter()
        .map(imu_link_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"attempted\":{},\"state_dof\":{},\"factor_count\":{},\"factor_rows\":{},\"prior_factor_rows\":{},\"visual_factor_rows\":{},\"imu_factor_rows\":{},\"bias_factor_rows\":{},\"landmark_count\":{},\"imu_link_count\":{},\"prior_rows\":{},\"prior_cost\":{},\"visual_cost\":{},\"imu_cost\":{},\"bias_cost\":{},\"status\":{},\"failure\":{},\"state_writeback\":{},\"landmark_writeback\":{},\"prior_carry\":{},\"imu_links\":[{}],\"lm\":[{}]}}",
        diagnostics.attempted,
        diagnostics.state_dof,
        diagnostics.factor_count,
        diagnostics.factor_rows,
        diagnostics.prior_factor_rows,
        diagnostics.visual_factor_rows,
        diagnostics.imu_factor_rows,
        diagnostics.bias_factor_rows,
        diagnostics.landmark_count,
        diagnostics.imu_link_count,
        diagnostics.prior_rows,
        json_number(diagnostics.prior_cost),
        json_number(diagnostics.visual_cost),
        json_number(diagnostics.imu_cost),
        json_number(diagnostics.bias_cost),
        json_string(&diagnostics.status),
        diagnostics
            .failure
            .as_deref().map_or_else(|| "null".into(), json_string),
        diagnostics.state_writeback,
        diagnostics.landmark_writeback,
        diagnostics.prior_carry,
        imu_links,
        lm,
    )
}

fn imu_link_json(link: &ImuLinkDiagnostics) -> String {
    format!(
        "{{\"from_frame_id\":{},\"to_frame_id\":{},\"dt\":{},\"r_p\":{},\"r_R\":{},\"r_v\":{},\"whitened_norm\":{},\"gyro_bias_delta\":{},\"accel_bias_delta\":{},\"bias_rotation_correction\":{},\"bias_velocity_correction\":{},\"bias_position_correction\":{},\"covariance_eigen_min\":{},\"covariance_eigen_max\":{}}}",
        link.from_frame_id,
        link.to_frame_id,
        json_number(link.delta_time),
        json_vec3(link.residual_position),
        json_vec3(link.residual_rotation),
        json_vec3(link.residual_velocity),
        json_number(link.whitened_norm),
        json_vec3(link.gyro_bias_delta),
        json_vec3(link.accel_bias_delta),
        json_vec3(link.bias_rotation_correction),
        json_vec3(link.bias_velocity_correction),
        json_vec3(link.bias_position_correction),
        json_number(link.covariance_eigen_min),
        json_number(link.covariance_eigen_max),
    )
}

fn lm_run_json(run: &LmRunDiagnostics) -> String {
    let trace = run
        .trace
        .iter()
        .map(|entry| {
            format!(
                "{{\"iteration\":{},\"lambda_before\":{},\"lambda_after\":{},\"cost_before\":{},\"model_cost\":{},\"actual_cost\":{},\"step_norm\":{},\"decision\":{}}}",
                entry.iteration,
                json_number(entry.lambda_before),
                json_number(entry.lambda_after),
                json_number(entry.cost_before),
                json_number(entry.model_cost),
                json_number(entry.actual_cost),
                json_number(entry.step_norm),
                json_string(&format!("{:?}", entry.decision)),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"pass\":{},\"iterations\":{},\"lambda\":{},\"initial_cost\":{},\"final_cost\":{},\"accepted\":{},\"rejected\":{},\"failure\":{},\"trace\":[{}]}}",
        json_string(&run.pass),
        run.iterations,
        json_number(run.lambda),
        json_number(run.initial_cost),
        json_number(run.final_cost),
        run.accepted,
        run.rejected,
        run.failure
            .as_deref().map_or_else(|| "null".into(), json_string),
        trace,
    )
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.17e}")
    } else {
        "null".into()
    }
}

fn json_vec3(value: [f64; 3]) -> String {
    format!(
        "[{}, {}, {}]",
        json_number(value[0]),
        json_number(value[1]),
        json_number(value[2]),
    )
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

fn json_u64_array(values: &[u64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_array_i64(values: &[i64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn aom_order_json(blocks: &[AomBlockData]) -> String {
    format!(
        "[{}]",
        blocks
            .iter()
            .map(|block| {
                format!(
                    "{{\"frame_id\":{},\"offset\":{},\"dof\":{},\"kind\":{}}}",
                    block.frame_id,
                    block.offset,
                    block.dof,
                    json_string(&block.kind),
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    )
}

impl Args {
    fn parse<I>(arguments: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = std::ffi::OsString>,
    {
        let mut euroc_dir = None;
        let mut calibration = None;
        let mut config = PathBuf::from("configs/basalt/euroc_config.json");
        let mut out_dir = PathBuf::from("target/basalt_euroc_vio_demo");
        let mut max_frames = None;
        let mut no_trace = false;
        let mut no_marg_data = false;
        let mut retained_marg_diagnostics = false;
        let mut native_companion_binding = None;
        let mut pipeline = false;
        let mut pipeline_capacity = 4usize;
        let mut decode_threads = 3usize;
        let mut threads = None;
        let mut mono = false;
        let mut dump_tracks = None;
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let option = argument.to_string_lossy();
            match option.as_ref() {
                "--help" | "-h" => return Err(Self::usage()),
                "--euroc-dir" | "--dataset" => {
                    euroc_dir = Some(next_path(&mut arguments, &option)?);
                }
                "--calibration" => {
                    calibration = Some(next_path(&mut arguments, &option)?);
                }
                "--config" => {
                    config = next_path(&mut arguments, &option)?;
                }
                "--out-dir" => {
                    out_dir = next_path(&mut arguments, &option)?;
                }
                "--max-frames" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| format!("{option} requires a value"))?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --max-frames: {error}"))?;
                    if value == 0 {
                        return Err("--max-frames must be positive".into());
                    }
                    max_frames = Some(value);
                }
                "--no-trace" => no_trace = true,
                "--dump-tracks" => {
                    dump_tracks = Some(next_path(&mut arguments, &option)?);
                }
                "--no-marg-data" => no_marg_data = true,
                "--retained-marg-diagnostics" => retained_marg_diagnostics = true,
                "--native-companion-binding" => {
                    native_companion_binding = Some(next_path(&mut arguments, &option)?);
                }
                "--pipeline" => pipeline = true,
                "--mono" => mono = true,
                "--pipeline-capacity" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| format!("{option} requires a value"))?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --pipeline-capacity: {error}"))?;
                    if value == 0 {
                        return Err("--pipeline-capacity must be positive".into());
                    }
                    pipeline_capacity = value;
                }
                "--decode-threads" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| format!("{option} requires a value"))?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --decode-threads: {error}"))?;
                    if value == 0 {
                        return Err("--decode-threads must be positive".into());
                    }
                    decode_threads = value;
                }
                "--threads" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| format!("{option} requires a value"))?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --threads: {error}"))?;
                    if value == 0 {
                        return Err("--threads must be positive".into());
                    }
                    threads = Some(value);
                }
                unknown => return Err(format!("unknown option `{unknown}`\n\n{}", Self::usage())),
            }
        }
        let euroc_dir =
            euroc_dir.ok_or_else(|| format!("--euroc-dir is required\n\n{}", Self::usage()))?;
        let calibration =
            calibration.ok_or_else(|| format!("--calibration is required\n\n{}", Self::usage()))?;
        Ok(Self {
            euroc_dir,
            calibration,
            config,
            out_dir,
            max_frames,
            no_trace,
            no_marg_data,
            retained_marg_diagnostics,
            native_companion_binding,
            pipeline,
            pipeline_capacity,
            decode_threads,
            threads,
            mono,
            dump_tracks,
        })
    }

    fn usage() -> String {
        "usage: basalt_euroc_vio_demo --euroc-dir DIR --calibration FILE [--config FILE] [--out-dir DIR] [--max-frames N] [--no-trace] [--no-marg-data] [--retained-marg-diagnostics] [--native-companion-binding FILE] [--pipeline] [--pipeline-capacity N] [--decode-threads N] [--threads N] [--mono] [--dump-tracks FILE]".into()
    }
}

fn next_path<I>(arguments: &mut I, option: &str) -> Result<PathBuf, String>
where
    I: Iterator<Item = std::ffi::OsString>,
{
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{option} requires a path"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Args {
        Args::parse(arguments.iter().map(std::ffi::OsString::from)).unwrap()
    }

    #[test]
    fn default_output_flags_preserve_existing_contract() {
        let args = parse(&["--euroc-dir", "dataset", "--calibration", "calib.json"]);
        assert!(!args.no_trace);
        assert!(!args.no_marg_data);
        assert!(args.native_companion_binding.is_none());
    }

    #[test]
    fn no_output_flags_are_independently_selectable() {
        let args = parse(&[
            "--euroc-dir",
            "dataset",
            "--calibration",
            "calib.json",
            "--no-trace",
            "--no-marg-data",
        ]);
        assert!(args.no_trace);
        assert!(args.no_marg_data);
    }

    #[test]
    fn adapter_dispatch_covers_all_output_flag_combinations() {
        let cases = [
            (false, false, AdapterProcessMode::Full, "full output"),
            (
                true,
                false,
                AdapterProcessMode::Full,
                "trace-only suppression keeps full adapter retention",
            ),
            (
                false,
                true,
                AdapterProcessMode::WithoutMargData,
                "no-MargData alone selects the lean MargData-suppressed path",
            ),
            (
                true,
                true,
                AdapterProcessMode::WithoutMargDataAndTrace,
                "both output classes suppressed",
            ),
        ];
        for (no_trace, no_marg_data, expected, description) in cases {
            let mut arguments = vec!["--euroc-dir", "dataset", "--calibration", "calib.json"];
            if no_trace {
                arguments.push("--no-trace");
            }
            if no_marg_data {
                arguments.push("--no-marg-data");
            }
            let args = parse(&arguments);
            assert_eq!(args.no_trace, no_trace, "{description}");
            assert_eq!(args.no_marg_data, no_marg_data, "{description}");
            assert_eq!(
                adapter_process_mode(args.no_trace, args.no_marg_data),
                expected,
                "{description}"
            );
        }
    }

    #[test]
    fn usage_documents_no_output_flags() {
        let usage = Args::usage();
        assert!(usage.contains("--no-trace"));
        assert!(usage.contains("--no-marg-data"));
        assert!(usage.contains("--native-companion-binding"));
    }

    #[test]
    fn native_companion_binding_path_is_parsed() {
        let args = parse(&[
            "--euroc-dir",
            "dataset",
            "--calibration",
            "calib.json",
            "--native-companion-binding",
            "identity.json",
        ]);
        assert_eq!(
            args.native_companion_binding,
            Some(PathBuf::from("identity.json"))
        );
    }

    #[test]
    fn pipeline_flags_default_off_and_are_parsed() {
        let args = parse(&["--euroc-dir", "dataset", "--calibration", "calib.json"]);
        assert!(!args.pipeline);
        assert_eq!(args.pipeline_capacity, 4);

        let args = parse(&[
            "--euroc-dir",
            "dataset",
            "--calibration",
            "calib.json",
            "--pipeline",
            "--pipeline-capacity",
            "8",
        ]);
        assert!(args.pipeline);
        assert_eq!(args.pipeline_capacity, 8);
    }

    #[test]
    fn decode_threads_defaults_to_three_and_is_parsed() {
        let args = parse(&["--euroc-dir", "dataset", "--calibration", "calib.json"]);
        assert_eq!(args.decode_threads, 3);

        let args = parse(&[
            "--euroc-dir",
            "dataset",
            "--calibration",
            "calib.json",
            "--decode-threads",
            "2",
        ]);
        assert_eq!(args.decode_threads, 2);
    }

    #[test]
    fn usage_documents_pipeline_flags() {
        let usage = Args::usage();
        assert!(usage.contains("--pipeline"));
        assert!(usage.contains("--pipeline-capacity"));
        assert!(usage.contains("--decode-threads"));
        assert!(usage.contains("--threads"));
    }

    #[test]
    fn mono_flag_defaults_off_and_is_parsed() {
        let args = parse(&["--euroc-dir", "dataset", "--calibration", "calib.json"]);
        assert!(!args.mono);

        let args = parse(&[
            "--euroc-dir",
            "dataset",
            "--calibration",
            "calib.json",
            "--mono",
        ]);
        assert!(args.mono);
        assert!(Args::usage().contains("--mono"));
    }

    #[test]
    fn threads_flag_defaults_unset_and_is_parsed() {
        let args = parse(&["--euroc-dir", "dataset", "--calibration", "calib.json"]);
        assert_eq!(args.threads, None);

        let args = parse(&[
            "--euroc-dir",
            "dataset",
            "--calibration",
            "calib.json",
            "--threads",
            "6",
        ]);
        assert_eq!(args.threads, Some(6));
    }
}
