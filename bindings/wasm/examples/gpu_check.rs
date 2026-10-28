//! Checks the GPU frontend (`src/gpu_flow.rs`) against the CPU one
//! (`DirectKltStream`) on a recording: each frame, the GPU starts from the
//! CPU's tracks, so every point is tracked from the same place on the same
//! images, and the two outcomes are compared. Then the corners each found
//! in the cells both left empty. Also times both. Then two sessions on the
//! GPU frontend, one frame by frame and one pipelined as the page's worker
//! runs it (the next frame on the GPU while the estimator works), which
//! must give the same poses.
//!
//! ```text
//! cargo run --release --example gpu_check -- <file.vrec> [--frames N]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use visloc_basalt::{DirectKltStream, RawU16Image, StereoFrame};
use visloc_wasm::gpu_flow::{GpuContext, GpuFlow};
use visloc_wasm::recording::decode;
use visloc_wasm::vio::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("usage: gpu_check <file.vrec> [--frames N]")?,
    );
    let mut max_frames = usize::MAX;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => max_frames = args.next().ok_or("--frames needs a value")?.parse()?,
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let rec = decode(&std::fs::read(&path)?)?;
    let rig = rec.rig.clone();
    let (calibration, config) = Session::frontend_config(&rig, &[])?;
    let (w, h) = (rig.width as usize, rig.height as usize);
    let mut cpu = DirectKltStream::new(calibration, config)?;
    let ctx = GpuContext::new_blocking()?;
    println!("GPU: {}", ctx.description);
    let mut gpu = GpuFlow::new(ctx, w, h, config)?;

    let (mut same, mut both_lost, mut cpu_only, mut gpu_only) = (0usize, 0usize, 0usize, 0usize);
    let (mut same_reason, mut lost_pairs) = (0usize, 0usize);
    let mut errors: Vec<f32> = Vec::new();
    let (mut corners_same, mut corners_differ, mut corners_compared) = (0usize, 0usize, 0usize);
    let (mut cpu_ms, mut gpu_ms) = (Vec::new(), Vec::new());
    let n = rec.frame_times_ns.len().min(max_frames);
    for i in 0..n {
        let t = rec.frame_times_ns[i];
        let pixels: Vec<u16> = rec.frame(i).iter().map(|&v| u16::from(v) << 8).collect();
        let before: Vec<_> = cpu.cam0_tracks().collect();
        gpu.set_tracks(before.iter().copied(), cpu.next_track_id());

        let tic = Instant::now();
        let image = RawU16Image::new(w, h, pixels.clone())?;
        let cpu_out = cpu.process_frame(StereoFrame::new(i as u64, t, image, None))?;
        cpu_ms.push(tic.elapsed().as_secs_f64() * 1e3);

        let tic = Instant::now();
        gpu.submit(i as u64, t, &pixels, |_| {})?;
        gpu.wait();
        let gpu_out = gpu.finish()?;
        gpu_ms.push(tic.elapsed().as_secs_f64() * 1e3);

        // The same input tracks, tracked by both.
        let cpu_pos: std::collections::BTreeMap<_, _> = cpu_out
            .observations
            .iter()
            .map(|o| (o.track_id, o.pixel))
            .collect();
        let gpu_pos: std::collections::BTreeMap<_, _> = gpu_out
            .observations
            .iter()
            .map(|o| (o.track_id, o.pixel))
            .collect();
        for (id, _, _) in &before {
            match (cpu_pos.get(id), gpu_pos.get(id)) {
                (Some(a), Some(b)) => {
                    same += 1;
                    errors.push((a - b).norm() as f32);
                }
                (None, None) => both_lost += 1,
                (Some(_), None) => cpu_only += 1,
                (None, Some(_)) => gpu_only += 1,
            }
        }
        let cpu_rejects: Vec<_> = cpu_out.reject_counters.iter().collect();
        for (reason, count) in &cpu_rejects {
            let g = gpu_out.reject_counters.count(**reason);
            same_reason += g.min(**count);
            lost_pairs += **count;
        }
        // Corners: compare the cells where both made a new track.
        let cell = config.fast.cell_size;
        let x_start = (w % cell) / 2;
        let y_start = (h % cell) / 2;
        let cell_of = |p: &nalgebra::Point2<f64>| {
            (
                ((p.x as usize).saturating_sub(x_start)) / cell,
                ((p.y as usize).saturating_sub(y_start)) / cell,
            )
        };
        let new_cpu: std::collections::BTreeMap<_, _> = cpu_out
            .created_track_ids
            .iter()
            .map(|id| (cell_of(&cpu_pos[id]), cpu_pos[id]))
            .collect();
        for id in &gpu_out.created_track_ids {
            let p = gpu_pos[id];
            if let Some(q) = new_cpu.get(&cell_of(&p)) {
                corners_compared += 1;
                if (p - q).norm() < 1e-6 {
                    corners_same += 1;
                } else {
                    corners_differ += 1;
                }
            }
        }
    }
    errors.sort_by(f32::total_cmp);
    let pct = |q: f64| {
        errors
            .get(((errors.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let total = same + both_lost + cpu_only + gpu_only;
    println!("{n} frames, {total} point-frames tracked from the same input");
    println!(
        "  kept by both {same}, lost by both {both_lost}, kept by the CPU only {cpu_only}, by the GPU only {gpu_only}"
    );
    println!(
        "  position difference when both kept: median {:.5} px, p99 {:.4}, max {:.3}",
        pct(0.5),
        pct(0.99),
        pct(1.0)
    );
    println!("  losses with the same reason counted on both: {same_reason} of {lost_pairs}");
    println!(
        "  corners in cells both filled: {corners_compared}, identical {corners_same}, different {corners_differ}"
    );
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "  time per frame: CPU {:.2} ms, GPU (submit to results) {:.2} ms",
        mean(&cpu_ms),
        mean(&gpu_ms)
    );
    pipeline_check(&rec, n)?;
    Ok(())
}

/// Serial and pipelined GPU sessions on the first `n` frames (no watchdog
/// or still start: a restart drops the frame on the GPU in the pipeline).
fn pipeline_check(
    rec: &visloc_wasm::recording::Recording,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let session = || -> Result<Session, Box<dyn std::error::Error>> {
        let mut s = Session::new(rec.rig.clone(), Vec::new())?;
        s.watchdog.enabled = false;
        s.still_start.enabled = false;
        s.attach_gpu(GpuContext::new_blocking()?)?;
        Ok(s)
    };
    let (mut serial, mut pipelined) = (session()?, session()?);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let (mut next_a, mut next_b) = (0usize, 0usize);
    let (mut ms_a, mut ms_b) = (0.0f64, 0.0f64);
    for i in 0..n {
        let t = rec.frame_times_ns[i];
        for (s, next) in [(&mut serial, &mut next_a), (&mut pipelined, &mut next_b)] {
            while *next < rec.imu.len() && !s.ready(t) {
                s.push_imu(rec.imu[*next]);
                *next += 1;
            }
        }
        if !serial.ready(t) {
            break;
        }
        let tic = Instant::now();
        serial.submit_gpu(t, rec.frame(i), |_| {})?;
        serial.gpu().expect("attached").wait();
        a.push(serial.finish_gpu()?);
        ms_a += tic.elapsed().as_secs_f64() * 1e3;
        let tic = Instant::now();
        if let Some(u) = pipelined.process_gpu_pipelined(t, rec.frame(i))? {
            b.push(u);
        }
        ms_b += tic.elapsed().as_secs_f64() * 1e3;
    }
    if let Some(u) = pipelined.flush_gpu()? {
        b.push(u);
    }
    let same = a.len() == b.len()
        && a.iter().zip(&b).all(|(x, y)| {
            x.timestamp_ns == y.timestamp_ns
                && x.camera_to_world.translation == y.camera_to_world.translation
                && x.camera_to_world.rotation == y.camera_to_world.rotation
        });
    println!(
        "pipelined vs frame-by-frame sessions: {} and {} frames, poses {}; {:.2} vs {:.2} ms a frame",
        b.len(),
        a.len(),
        if same { "identical" } else { "DIFFERENT" },
        ms_b / b.len().max(1) as f64,
        ms_a / a.len().max(1) as f64
    );
    Ok(())
}
