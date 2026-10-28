//! Replays an ADVIO iPhone recording through the browser's VIO session, as a
//! phone browser would feed it: frames decimated from 60 to `--fps`, scaled
//! to `--width`, grey; accelerometer and gyroscope resampled onto one
//! `--imu-hz` timeline (a browser delivers them together in each
//! `devicemotion` event). Prints the per-frame cost and the trajectory error
//! against ADVIO's ground truth (and ARKit's, for reference) after SE(3) and
//! Sim(3) alignment.
//!
//! ```text
//! cargo run --release --example advio -- <advio-NN dir> [--fps 30] [--imu-hz 60] \
//!     [--width 360] [--start 0] [--seconds 60] [--td-ms 0] [--noise-scale 1] \
//!     [--bias-scale S] [--pinhole] [--print-every N] [--set config.key=json ...] \
//!     [--out traj.csv] [--no-still-start] [--gpu] [--estimate-every 2]
//! ```
//!
//! ADVIO (Cortés et al., ECCV 2018) is CC BY-NC 4.0. Its frame timestamps
//! run about 300 ms behind its IMU clock (cross-correlating ARKit's and the
//! gyroscope's rotation rates: -300 ms on sequences 1 and 2, -312 on 13,
//! -328 on 16), hence `--td-ms`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion, Vector3};
use visloc_basalt::ImuSample;
use visloc_core::geometry::SE3;
use visloc_wasm::vio::{PhoneRig, Session};

/// ADVIO's per-batch iPhone calibrations (`calibration/README.md`): the
/// first sequence of the batch, pinhole at 720x1280, radial-tangential
/// `[k1, k2, p1, p2]`. All batches share Kalibr's `T_cam_imu`.
const BATCHES: [(u32, [f64; 4], [f64; 4]); 4] = [
    (
        1,
        [1077.2, 1079.3, 362.145, 636.3873],
        [0.0478, 0.0339, -0.00033, -0.00091],
    ),
    (
        13,
        [1082.4, 1084.4, 364.68, 643.31],
        [0.0366, 0.0803, 0.0007, -0.0002],
    ),
    (
        18,
        [1076.9, 1078.5, 360.96, 639.31],
        [0.0510, -0.0354, -0.0054, 0.0473],
    ),
    (
        20,
        [1081.1, 1082.1, 359.59, 640.79],
        [0.0556, -0.0454, 0.0009, -0.0018],
    ),
];
const T_CAM_IMU: [[f64; 4]; 3] = [
    [
        0.9999763379093255,
        -0.004079205042965442,
        -0.005539287650170447,
        -0.008977668364731128,
    ],
    [
        -0.004066386342107199,
        -0.9999890330121858,
        0.0023234365646622014,
        0.07557012320238939,
    ],
    [
        -0.00554870467502187,
        -0.0023008567036498766,
        -0.9999819588046867,
        -0.005545773942541918,
    ],
];

fn read_csv(path: &Path) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .collect())
}

/// Linear interpolation of rows `(t, v...)` at `t`.
fn interp(rows: &[Vec<f64>], t: f64) -> Option<Vec<f64>> {
    let k = rows.partition_point(|r| r[0] < t);
    if k == 0 || k >= rows.len() {
        return None;
    }
    let (a, b) = (&rows[k - 1], &rows[k]);
    let w = (t - a[0]) / (b[0] - a[0]);
    Some((1..a.len()).map(|i| a[i] + w * (b[i] - a[i])).collect())
}

/// Umeyama alignment of `est` onto `gt`: the RMSE after the best rigid
/// (`scale = false`) or similarity transform, and the scale.
fn aligned_rmse(est: &[Vector3<f64>], gt: &[Vector3<f64>], with_scale: bool) -> (f64, f64) {
    let n = est.len() as f64;
    let me = est.iter().sum::<Vector3<f64>>() / n;
    let mg = gt.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    let mut var = 0.0;
    for (e, g) in est.iter().zip(gt) {
        cov += (g - mg) * (e - me).transpose();
        var += (e - me).norm_squared();
    }
    cov /= n;
    var /= n;
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = u * d * vt;
    let s = if with_scale {
        (svd.singular_values.component_mul(&d.diagonal())).sum() / var
    } else {
        1.0
    };
    let t = mg - s * r * me;
    let sq: f64 = est
        .iter()
        .zip(gt)
        .map(|(e, g)| (s * r * e + t - g).norm_squared())
        .sum();
    ((sq / n).sqrt(), s)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: advio <advio-NN dir> [options]")?);
    let (mut fps, mut imu_hz, mut width) = (30.0f64, 60.0f64, 360u32);
    let (mut start, mut seconds, mut td_ms, mut noise_scale) =
        (0.0f64, f64::INFINITY, 0.0f64, 1.0f64);
    let mut overrides: Vec<(String, serde_json::Value)> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut print_every = 0u64;
    let mut bias_scale: Option<f64> = None;
    let mut pinhole = false;
    let mut tracks_out: Option<PathBuf> = None;
    let mut record: Option<PathBuf> = None;
    let mut focal_scale = 1.0f64;
    let mut still_start = true;
    let mut gpu = false;
    let mut estimate_every: Option<u32> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--fps" => fps = next()?.parse()?,
            "--imu-hz" => imu_hz = next()?.parse()?,
            "--width" => width = next()?.parse()?,
            "--start" => start = next()?.parse()?,
            "--seconds" => seconds = next()?.parse()?,
            "--td-ms" => td_ms = next()?.parse()?,
            "--noise-scale" => noise_scale = next()?.parse()?,
            "--set" => {
                let kv = next()?;
                let (k, v) = kv.split_once('=').ok_or("--set key=json")?;
                overrides.push((k.to_string(), serde_json::from_str(v)?));
            }
            "--out" => out = Some(PathBuf::from(next()?)),
            "--print-every" => print_every = next()?.parse()?,
            "--bias-scale" => bias_scale = Some(next()?.parse()?),
            // Ignore the lens distortion (a phone browser's frames may have it).
            "--pinhole" => pinhole = true,
            // Every frame's tracks (`t,id,x,y` rows, undistorted pixels).
            "--tracks-out" => tracks_out = Some(PathBuf::from(next()?)),
            // Also write the frames and IMU samples fed to the session as a
            // `.vrec` recording (for `replay` and `web/node_replay.mjs`).
            "--record" => record = Some(PathBuf::from(next()?)),
            // Mis-scale the focal length (calibration-error sensitivity).
            "--focal-scale" => focal_scale = next()?.parse()?,
            // Start (and restart) without waiting for a still phone.
            "--no-still-start" => still_start = false,
            "--gpu" => gpu = true,
            "--estimate-every" => estimate_every = Some(next()?.parse()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let height = width * 16 / 9;
    let s = width as f64 / 720.0;
    let sequence: u32 = dir
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.rsplit('-').next())
        .and_then(|n| n.parse().ok())
        .ok_or("the directory must be named advio-NN")?;
    let &(_, [fx, fy, cx, cy], distortion) = BATCHES
        .iter()
        .rev()
        .find(|b| b.0 <= sequence)
        .expect("batch 1 covers every sequence");
    let distortion = if pinhole { [0.0; 4] } else { distortion };
    let t_cam_imu = Matrix4::from_fn(|r, c| {
        if r < 3 {
            T_CAM_IMU[r][c]
        } else if c == 3 {
            1.0
        } else {
            0.0
        }
    });
    let rot = Rotation3::from_matrix(&t_cam_imu.fixed_view::<3, 3>(0, 0).into_owned());
    let cam_from_imu = SE3::new(
        UnitQuaternion::from_rotation_matrix(&rot),
        t_cam_imu.fixed_view::<3, 1>(0, 3).into_owned(),
    );
    let rig = PhoneRig {
        width,
        height,
        fx: fx * s * focal_scale,
        fy: fy * s * focal_scale,
        cx: (cx + 0.5) * s - 0.5,
        cy: (cy + 0.5) * s - 0.5,
        distortion,
        imu_from_camera: cam_from_imu.inverse(),
        imu_rate_hz: imu_hz,
        accel_noise: 4.8e-3 * noise_scale,
        gyro_noise: 2.4e-3 * noise_scale,
        accel_bias_walk: 2.1e-4 * bias_scale.unwrap_or(noise_scale),
        gyro_bias_walk: 5.1e-5 * bias_scale.unwrap_or(noise_scale),
        frame_rate_hz: fps,
    };
    println!(
        "rig {width}x{height} f {:.1} {:.1} c {:.1} {:.1}; imu {imu_hz} Hz; camera {fps} fps; td {td_ms} ms",
        rig.fx, rig.fy, rig.cx, rig.cy
    );

    let iphone = dir.join("iphone");
    let accel = read_csv(&iphone.join("accelerometer.csv"))?;
    let gyro = read_csv(&iphone.join("gyro.csv"))?;
    let frames = read_csv(&iphone.join("frames.csv"))?;
    let gt = read_csv(&dir.join("ground-truth").join("pose.csv"))?;
    let arkit = read_csv(&iphone.join("arkit.csv"))?;
    let t0 = accel[0][0].max(gyro[0][0]);
    let t_end = accel.last().unwrap()[0].min(gyro.last().unwrap()[0]);
    let imu: Vec<ImuSample> = (0..)
        .map(|k| t0 + k as f64 / imu_hz)
        .take_while(|&t| t < t_end)
        .filter_map(|t| {
            let a = interp(&accel, t)?;
            let g = interp(&gyro, t)?;
            Some(ImuSample::new(
                (t * 1e9).round() as i64,
                Vector3::new(g[0], g[1], g[2]),
                Vector3::new(a[0], a[1], a[2]),
            ))
        })
        .collect();

    // Every `step`-th video frame, from `start` for `seconds`.
    let step = (60.0 / fps).round().max(1.0) as usize;
    let first = frames.partition_point(|r| r[0] < start.max(t0 + 0.1));
    let selected: Vec<usize> = (first..frames.len())
        .step_by(step)
        .take_while(|&n| frames[n][0] < start + seconds)
        .collect();
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-ss"])
        .arg(format!("{:.3}", frames[first][0] - frames[0][0]))
        .arg("-i")
        .arg(iphone.join("frames.mov"))
        .args([
            "-vf",
            &format!("select=not(mod(n\\,{step})),scale={width}:{height}:flags=area,format=gray"),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut video = child.stdout.take().unwrap();

    let mut session = Session::new(rig.clone(), overrides)?;
    session.still_start.enabled = still_start;
    if let Some(every) = estimate_every {
        session.estimate_every = every;
    }
    if gpu {
        let ctx = visloc_wasm::gpu_flow::GpuContext::new_blocking()?;
        println!("GPU frontend: {}", ctx.description);
        session.attach_gpu(ctx)?;
    }
    let mut buf = vec![0u8; (width * height) as usize];
    let mut next_imu = 0usize;
    let mut times_ms: Vec<f64> = Vec::new();
    // Estimated, ground-truth and ARKit positions at the frames that have all three.
    let (mut est, mut ref_gt, mut ref_arkit) = (Vec::new(), Vec::new(), Vec::new());
    let mut csv = String::from("t,x,y,z,qw,qx,qy,qz,tracks,connected,keyframe\n");
    let mut tracks_csv = String::new();
    let (mut rec_times, mut rec_frames, mut rec_imu) = (Vec::new(), Vec::new(), Vec::new());
    let started = Instant::now();
    let mut keyframes = 0usize;
    let mut tracking_frames = 0usize;
    // Frontend health, as replay reports it: landmarks in view, and why
    // tracks were dropped (as a share of the tracks carried into a frame).
    let (mut connected, mut coasting) = (Vec::new(), 0usize);
    let (mut rejects, mut tracked_on, mut last_tracks) = (
        std::collections::BTreeMap::<String, usize>::new(),
        0usize,
        0usize,
    );
    // Per track id: when its current run of consecutive frames began and
    // the frame it was last seen in (ids restart with each estimator).
    let mut track_runs = std::collections::HashMap::<u64, (i64, i64, usize)>::new();
    let mut lives: Vec<f64> = Vec::new();
    let mut frame_index = 0usize;
    for &n in &selected {
        if video.read_exact(&mut buf).is_err() {
            break;
        }
        let t = frames[n][0] + td_ms * 1e-3;
        let t_ns = (t * 1e9).round() as i64;
        while next_imu < imu.len() && !session.ready(t_ns) {
            session.push_imu(imu[next_imu]);
            if record.is_some() {
                rec_imu.push(imu[next_imu]);
            }
            next_imu += 1;
        }
        if record.is_some() {
            rec_times.push(t_ns);
            rec_frames.extend_from_slice(&buf);
        }
        if !session.ready(t_ns) {
            break;
        }
        let tic = Instant::now();
        let update = if gpu {
            session.submit_gpu(t_ns, &buf, |_| {})?;
            session.gpu().expect("attached").wait();
            session.finish_gpu()?
        } else {
            session.process(t_ns, &buf)?
        };
        times_ms.push(tic.elapsed().as_secs_f64() * 1e3);
        keyframes += usize::from(update.is_keyframe);
        tracking_frames += usize::from(update.status == visloc_wasm::vio::Status::Tracking);
        coasting += usize::from(update.status == visloc_wasm::vio::Status::Coasting);
        connected.push(update.connected);
        tracked_on += last_tracks;
        last_tracks = update.tracks.len();
        for (reason, k) in &update.track_rejects {
            *rejects.entry(reason.clone()).or_default() += k;
        }
        frame_index += 1;
        for &(id, _, _) in &update.tracks {
            let run = track_runs.entry(id).or_insert((t_ns, t_ns, frame_index));
            if run.2 + 1 < frame_index {
                lives.push((run.1 - run.0) as f64 * 1e-9);
                *run = (t_ns, t_ns, frame_index);
            }
            run.1 = t_ns;
            run.2 = frame_index;
        }
        if tracks_out.is_some() {
            for &(id, x, y) in &update.tracks {
                tracks_csv.push_str(&format!("{t_ns},{id},{x},{y}\n"));
            }
        }
        let p = update.camera_to_world.translation;
        let q = update.camera_to_world.rotation;
        csv.push_str(&format!(
            "{:.6},{},{},{},{},{},{},{},{},{},{}\n",
            frames[n][0],
            p.x,
            p.y,
            p.z,
            q.w,
            q.i,
            q.j,
            q.k,
            update.tracks.len(),
            update.connected,
            u8::from(update.is_keyframe)
        ));
        if !p.iter().all(|v| v.is_finite()) {
            println!("non-finite pose at frame {}", update.frame_id);
            break;
        }
        if let (Some(g), Some(a)) = (interp(&gt, frames[n][0]), interp(&arkit, frames[n][0])) {
            est.push(p);
            ref_gt.push(Vector3::new(g[0], g[1], g[2]));
            ref_arkit.push(Vector3::new(a[0], a[1], a[2]));
        }
        let every = if print_every > 0 {
            print_every
        } else {
            fps as u64 * 10
        };
        if update.frame_id > 0 && update.frame_id % (fps as u64 * 5) == 0 {
            if let Some(e) = session.time_sync() {
                println!(
                    "t {:7.2}s sync: offset {:+.1} ms corr {:.3} (rig {:.3}) rotation off by {:.1} deg (spread {:.2}) focal x{:.3} ({} intervals, {:.2} rad/s)",
                    frames[n][0],
                    e.offset_ms,
                    e.correlation,
                    e.correlation_assumed,
                    e.rotation_error_deg,
                    e.axis_spread,
                    e.focal_scale,
                    e.intervals,
                    e.rate_rad_s
                );
            }
        }
        if let visloc_wasm::vio::Status::Reset(reason) = update.status {
            println!("t {:7.2}s reset ({})", frames[n][0], reason.name());
        }
        if update.frame_id % every == 0 {
            println!(
                "t {:7.2}s frame {:5} tracks {:3} connected {:3} landmarks {:4} |v| {:.2} m/s ba {:.3} bg {:.4} kf {} ({:.1} ms)",
                frames[n][0],
                update.frame_id,
                update.tracks.len(),
                update.connected,
                session.landmarks().len(),
                update.state.velocity_world_m_s.norm(),
                update.state.accel_bias_m_s2.norm(),
                update.state.gyro_bias_rad_s.norm(),
                keyframes,
                times_ms.last().unwrap()
            );
        }
    }
    let _ = child.kill();
    let wall = started.elapsed().as_secs_f64();
    let mut sorted = times_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let processed = times_ms.len();
    println!(
        "{processed} frames ({keyframes} keyframes, {:.0}% tracking, {} resets) in {wall:.1} s; per frame mean {:.1} ms, p50 {:.1}, p95 {:.1}, max {:.1}",
        100.0 * tracking_frames as f64 / processed.max(1) as f64,
        session.resets(),
        times_ms.iter().sum::<f64>() / processed.max(1) as f64,
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95),
        sorted.last().copied().unwrap_or(0.0)
    );
    let thin = connected.iter().filter(|&&c| c < 10).count();
    connected.sort_unstable();
    // Track life: first to last frame of each run, over runs longer than
    // a frame.
    lives.extend(track_runs.values().map(|run| (run.1 - run.0) as f64 * 1e-9));
    lives.retain(|&life| life > 0.0);
    lives.sort_by(f64::total_cmp);
    println!(
        "landmarks in view: median {}, under 10 in {:.0}% of frames; coasting {:.1}%; track life median {:.2} s, p90 {:.2} s",
        connected.get(connected.len() / 2).copied().unwrap_or(0),
        100.0 * thin as f64 / processed.max(1) as f64,
        100.0 * coasting as f64 / processed.max(1) as f64,
        lives.get(lives.len() / 2).copied().unwrap_or(0.0),
        lives.get(lives.len() * 9 / 10).copied().unwrap_or(0.0),
    );
    let dropped: Vec<String> = rejects
        .iter()
        .map(|(reason, k)| {
            format!(
                "{reason} {:.1}%",
                100.0 * *k as f64 / tracked_on.max(1) as f64
            )
        })
        .collect();
    println!("tracks dropped per frame: {}", dropped.join(", "));
    if est.len() > 10 {
        let path: f64 = ref_gt.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
        let (se3, _) = aligned_rmse(&est, &ref_gt, false);
        let (sim3, scale) = aligned_rmse(&est, &ref_gt, true);
        println!(
            "vs ground truth ({path:.1} m path): ATE SE(3) {se3:.3} m, Sim(3) {sim3:.3} m at scale {scale:.3}"
        );
        let (ark_se3, _) = aligned_rmse(&ref_arkit, &ref_gt, false);
        let (vs_ark, _) = aligned_rmse(&est, &ref_arkit, false);
        println!("ARKit vs ground truth: ATE SE(3) {ark_se3:.3} m; this vs ARKit: {vs_ark:.3} m");
    }
    if let Some(out) = out {
        std::fs::write(out, csv)?;
    }
    if let Some(path) = tracks_out {
        std::fs::write(path, tracks_csv)?;
    }
    if let Some(path) = record {
        // The session consumed frames up to the last one; drop any frame it
        // could not process.
        rec_times.truncate(processed);
        rec_frames.truncate(processed * (width * height) as usize);
        let extra = serde_json::json!({ "source": dir.display().to_string(), "tdMs": td_ms });
        std::fs::write(
            path,
            visloc_wasm::recording::encode(&rig, &rec_times, &rec_frames, &rec_imu, extra),
        )?;
    }
    Ok(())
}
