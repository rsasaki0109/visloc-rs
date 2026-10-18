//! Replays a `.vrec` recording (captured by the demo page on a phone, or
//! written by `advio --record`) through the VIO session natively, printing
//! the per-frame cost, the watchdog's resets, the travelled path, and how
//! well the map held up (landmarks in view, frames tracking), and why the
//! frontend dropped tracks (as a share of the tracks carried into a frame):
//! the measures to tune a phone capture by, which has no ground truth.
//!
//! ```text
//! cargo run --release --example replay -- <file.vrec> [--td-ms 0] [--noise-scale 1] \
//!     [--bias-scale 1] [--fx F] [--no-watchdog] [--no-still-start] \
//!     [--set config.key=json ...] [--print-every 30] [--out traj.csv] \
//!     [--tracks-out tracks.csv] [--skip-s 0.5] [--max-s 3] [--every 2] [--gpu] [--estimate-every 2] [--max-speed 3]
//! ```
//!
//! `--bias-scale` scales only the bias random walks (on top of
//! `--noise-scale`). `--skip-s` starts tracking that many seconds in (the
//! motion samples before still reach the session, as on the page, where the
//! sensors run before tracking starts); `--max-s` stops that many seconds
//! later. `--every 2` tracks every other frame (a 60 fps recording at 30 Hz,
//! as the page tracks when filming at 60 fps without tracking at 60 Hz).
//! `--no-still-start` starts without waiting for a still phone.
//! `--gpu` runs the frontend on the GPU (`src/gpu_flow.rs`), as the page can.
//! `--out` writes one row per frame (pose, velocity, biases, tracks,
//! landmarks in view, status); `--tracks-out` one row per tracked point
//! (`frame,t_ns,track_id,x,y`, pixels on the recording's frames).
//!
//! `--td-ms` shifts the frame timestamps (camera time + td = IMU time),
//! `--noise-scale` scales the recorded rig's IMU noise, `--fx` replaces its
//! focal length (fy scaled alike): the calibration knobs to try on a phone
//! capture.

use std::path::PathBuf;
use std::time::Instant;

use visloc_wasm::recording;
use visloc_wasm::vio::{Session, Status};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or("usage: replay <file.vrec> [options]")?);
    let (mut td_ms, mut noise_scale, mut fx, mut watchdog, mut every) =
        (0.0f64, 1.0f64, None, true, 30u64);
    let mut overrides: Vec<(String, serde_json::Value)> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut tracks_out: Option<PathBuf> = None;
    let mut skip_s = 0.0f64;
    let mut bias_scale = 1.0f64;
    let mut max_s = f64::INFINITY;
    let mut step = 1usize;
    let mut still_start = true;
    let mut gpu = false;
    let mut estimate_every: Option<u32> = None;
    let mut max_speed: Option<f64> = None;
    let mut gpu_split = [0.0f64; 3];
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--td-ms" => td_ms = next()?.parse()?,
            "--noise-scale" => noise_scale = next()?.parse()?,
            "--fx" => fx = Some(next()?.parse::<f64>()?),
            "--no-watchdog" => watchdog = false,
            "--no-still-start" => still_start = false,
            "--gpu" => gpu = true,
            "--estimate-every" => estimate_every = Some(next()?.parse()?),
            "--max-speed" => max_speed = Some(next()?.parse()?),
            "--print-every" => every = next()?.parse()?,
            "--set" => {
                let kv = next()?;
                let (k, v) = kv.split_once('=').ok_or("--set key=json")?;
                overrides.push((k.to_string(), serde_json::from_str(v)?));
            }
            "--out" => out = Some(PathBuf::from(next()?)),
            "--tracks-out" => tracks_out = Some(PathBuf::from(next()?)),
            "--skip-s" => skip_s = next()?.parse()?,
            "--max-s" => max_s = next()?.parse()?,
            "--every" => step = next()?.parse::<usize>()?.max(1),
            "--bias-scale" => bias_scale = next()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let rec = recording::decode(&std::fs::read(&path)?)?;
    let mut rig = rec.rig.clone();
    // A recording of every camera frame replays at the camera's rate (or a
    // fraction of it with --every).
    if let Some(hz) = rec.header.get("frameRateHz").and_then(|v| v.as_f64()) {
        rig.frame_rate_hz = if hz / step as f64 > 45.0 { 60.0 } else { 30.0 };
    }
    rig.accel_noise *= noise_scale;
    rig.gyro_noise *= noise_scale;
    rig.accel_bias_walk *= noise_scale * bias_scale;
    rig.gyro_bias_walk *= noise_scale * bias_scale;
    if let Some(fx) = fx {
        rig.fy *= fx / rig.fx;
        rig.fx = fx;
    }
    println!(
        "{}: {} frames {}x{}, {} IMU samples; f {:.1} {:.1} c {:.1} {:.1}; td {td_ms} ms",
        path.display(),
        rec.frame_times_ns.len(),
        rig.width,
        rig.height,
        rec.imu.len(),
        rig.fx,
        rig.fy,
        rig.cx,
        rig.cy
    );
    if let Some(device) = rec.header.get("device") {
        println!("device: {device}");
    }
    if let Some(hz) = rec.header.get("frameRateHz") {
        let live = rec
            .header
            .get("trackedEvery")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        println!("every camera frame recorded, {hz} fps; the page tracked every {live}");
    }
    let mut session = Session::new(rig, overrides)?;
    session.watchdog.enabled = watchdog;
    session.still_start.enabled = still_start;
    if let Some(every) = estimate_every {
        session.estimate_every = every;
    }
    if let Some(speed) = max_speed {
        session.watchdog.max_speed_m_s = speed;
    }
    if gpu {
        let ctx = visloc_wasm::gpu_flow::GpuContext::new_blocking()?;
        println!("GPU frontend: {}", ctx.description);
        session.attach_gpu(ctx)?;
    }
    let td_ns = (td_ms * 1e6).round() as i64;
    let mut next_imu = 0usize;
    let mut times = Vec::new();
    let (mut connected, mut tracks, mut tracking) = (Vec::new(), Vec::new(), 0usize);
    let (mut coasting, mut coast_run, mut longest_coast) = (0usize, 0usize, 0usize);
    let mut csv = String::from(
        "t,x,y,z,qw,qx,qy,qz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,tracks,connected,imu_samples,keyframe,status\n",
    );
    let mut tracks_csv = String::from("frame,t_ns,track_id,x,y\n");
    let mut rejects = std::collections::BTreeMap::<String, usize>::new();
    let mut tracked_on = 0usize;
    let (mut path_len, mut last): (f64, Option<nalgebra::Vector3<f64>>) = (0.0, None);
    let started = Instant::now();
    let first_frame_ns = rec.frame_times_ns.first().copied().unwrap_or(0);
    for (i, &t) in rec.frame_times_ns.iter().enumerate() {
        // Frames in the first `--skip-s` are not tracked; their IMU samples
        // still reach the session with the first tracked frame.
        if ((t - first_frame_ns) as f64) < skip_s * 1e9 {
            continue;
        }
        if ((t - first_frame_ns) as f64) > (skip_s + max_s) * 1e9 {
            break;
        }
        if i % step != 0 {
            continue;
        }
        let t = t + td_ns;
        while next_imu < rec.imu.len() && !session.ready(t) {
            session.push_imu(rec.imu[next_imu]);
            next_imu += 1;
        }
        if !session.ready(t) {
            break;
        }
        let tic = Instant::now();
        let u = if gpu {
            let a = Instant::now();
            session.submit_gpu(t, rec.frame(i), |_| {})?;
            let b = Instant::now();
            session.gpu().expect("attached").wait();
            let c = Instant::now();
            let u = session.finish_gpu()?;
            gpu_split[0] += (b - a).as_secs_f64() * 1e3;
            gpu_split[1] += (c - b).as_secs_f64() * 1e3;
            gpu_split[2] += c.elapsed().as_secs_f64() * 1e3;
            u
        } else {
            session.process(t, rec.frame(i))?
        };
        times.push(tic.elapsed().as_secs_f64() * 1e3);
        connected.push(u.connected);
        tracked_on += tracks.last().copied().unwrap_or(0);
        tracks.push(u.tracks.len());
        for (reason, n) in &u.track_rejects {
            *rejects.entry(reason.clone()).or_default() += n;
        }
        tracking += usize::from(matches!(u.status, Status::Tracking));
        if matches!(u.status, Status::Coasting) {
            coasting += 1;
            coast_run += 1;
            longest_coast = longest_coast.max(coast_run);
        } else {
            coast_run = 0;
        }
        let p = u.camera_to_world.translation;
        if let Some(l) = last {
            path_len += (p - l).norm();
        }
        last = Some(p);
        let q = u.camera_to_world.rotation;
        let status = match u.status {
            Status::Waiting => "wait".to_string(),
            Status::Initializing => "init".to_string(),
            Status::Tracking => "track".to_string(),
            Status::Coasting => "coast".to_string(),
            Status::Reset(r) => format!("reset-{}", r.name()),
        };
        if let Status::Reset(r) = u.status {
            println!("frame {i}: reset ({})", r.name());
        }
        if every > 0 && (i as u64).is_multiple_of(every) {
            println!(
                "frame {i:5} tracks {:3} connected {:3} {status:6} p ({:.2}, {:.2}, {:.2}) |v| {:.2} ba {:.3} bg {:.4} ({:.1} ms)",
                u.tracks.len(),
                u.connected,
                p.x,
                p.y,
                p.z,
                u.state.velocity_world_m_s.norm(),
                u.state.accel_bias_m_s2.norm(),
                u.state.gyro_bias_rad_s.norm(),
                times.last().unwrap()
            );
        }
        let (v, bg, ba) = (
            u.state.velocity_world_m_s,
            u.state.gyro_bias_rad_s,
            u.state.accel_bias_m_s2,
        );
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{status}\n",
            u.timestamp_ns,
            p.x,
            p.y,
            p.z,
            q.w,
            q.i,
            q.j,
            q.k,
            v.x,
            v.y,
            v.z,
            bg.x,
            bg.y,
            bg.z,
            ba.x,
            ba.y,
            ba.z,
            u.tracks.len(),
            u.connected,
            u.imu_samples,
            u8::from(u.is_keyframe)
        ));
        if tracks_out.is_some() {
            for &(id, x, y) in &u.tracks {
                tracks_csv.push_str(&format!("{i},{},{id},{x},{y}\n", u.timestamp_ns));
            }
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let n = times.len().max(1);
    println!(
        "{} frames in {:.1} s, {} resets; per frame mean {:.1} ms, p50 {:.1}, p95 {:.1}; path {:.2} m; final p {:?}",
        times.len(),
        started.elapsed().as_secs_f64(),
        session.resets(),
        times.iter().sum::<f64>() / n as f64,
        sorted.get(n / 2).copied().unwrap_or(0.0),
        sorted.get(n * 95 / 100).copied().unwrap_or(0.0),
        path_len,
        last.map(|p| [p.x, p.y, p.z])
    );
    if gpu {
        let k = n.max(1) as f64;
        println!(
            "GPU frontend per frame: submit {:.2} ms, wait {:.2} ms, finish (estimator) {:.2} ms",
            gpu_split[0] / k,
            gpu_split[1] / k,
            gpu_split[2] / k
        );
    }
    let median = |v: &mut Vec<usize>| {
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0)
    };
    let thin = connected.iter().filter(|&&c| c < 10).count();
    println!(
        "landmarks in view: median {}, under 10 in {:.0}% of frames; tracks median {}; tracking {:.0}% of frames, coasting {:.1}% (longest {longest_coast} frames)",
        median(&mut connected.clone()),
        100.0 * thin as f64 / n as f64,
        median(&mut tracks),
        100.0 * tracking as f64 / n as f64,
        100.0 * coasting as f64 / n as f64
    );
    let dropped: Vec<String> = rejects
        .iter()
        .map(|(reason, n)| {
            format!(
                "{reason} {:.1}%",
                100.0 * *n as f64 / tracked_on.max(1) as f64
            )
        })
        .collect();
    println!("tracks dropped per frame: {}", dropped.join(", "));
    if let Some(path) = tracks_out {
        std::fs::write(path, tracks_csv)?;
    }
    if let Some(out) = out {
        std::fs::write(out, csv)?;
    }
    Ok(())
}
