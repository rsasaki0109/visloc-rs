//! Compare GPU SIFT with the CPU `extract_sift` on a directory of images:
//! wall time, keypoint agreement and descriptor similarity.
//!
//! cargo run --release -p visloc-sift-gpu --features gpu --example sift_gpu_bench -- \
//!     --images <dir> [--max-images 5] [--max-keypoints N] [--no-cpu] [--workers N]
//! Use `--max-keypoints usize::MAX` to benchmark the unlimited library default.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::Instant;

use visloc_sift_gpu::{GpuContext, SiftGpu};
use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig, SiftKeypoint};

const DEFAULT_MAX_KEYPOINTS: usize = 8192;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn key(k: &SiftKeypoint) -> (i64, i64, i64) {
    (
        (k.x * 8.0).round() as i64,
        (k.y * 8.0).round() as i64,
        (k.sigma * 1000.0).round() as i64,
    )
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn percentile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() as f64 - 1.0) * q).ceil() as usize;
    sorted.get(index).copied()
}

fn emit_json_number(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), |v| format!("{v:.6}"))
}

fn run_worker_bench(paths: &[PathBuf], max_keypoints: usize, workers: usize, jsonl: bool) {
    assert!(!paths.is_empty(), "no benchmark images found");
    let next = AtomicUsize::new(0);
    let start_gate = Barrier::new(workers + 1);
    let results: Vec<Mutex<Option<(f64, usize, usize, usize)>>> =
        (0..paths.len()).map(|_| Mutex::new(None)).collect();
    let config = SiftConfig {
        max_keypoints,
        ..SiftConfig::default()
    };
    let adapter_name = GpuContext::new().expect("gpu").adapter_info.name;
    let mut wall_start = None;
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let next = &next;
            let results = &results;
            let start_gate = &start_gate;
            let config = &config;
            scope.spawn(move || {
                let mut gpu = SiftGpu::new(GpuContext::new().expect("gpu worker"));
                // Warm every device/context before the timed work queue.
                let warm = image::open(&paths[0]).unwrap().to_luma8();
                let (ww, wh) = (warm.width() as usize, warm.height() as usize);
                let warm_px: Vec<f32> = warm.as_raw().iter().map(|&b| b as f32).collect();
                let warm_img = GrayImage::new(ww, wh, &warm_px).unwrap();
                let _ = gpu.extract(&warm_img, config).unwrap();
                start_gate.wait();

                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= paths.len() {
                        break;
                    }
                    let gray = image::open(&paths[index]).unwrap().to_luma8();
                    let (w, h) = (gray.width() as usize, gray.height() as usize);
                    let pixels: Vec<f32> = gray.as_raw().iter().map(|&b| b as f32).collect();
                    let image = GrayImage::new(w, h, &pixels).unwrap();
                    let start = Instant::now();
                    let (keypoints, _) = gpu.extract(&image, config).unwrap();
                    let elapsed = start.elapsed().as_secs_f64();
                    *results[index].lock().expect("result lock") =
                        Some((elapsed, keypoints.len(), w, h));
                }
            });
        }
        start_gate.wait();
        wall_start = Some(Instant::now());
    });
    let wall_s = wall_start
        .expect("worker barrier started")
        .elapsed()
        .as_secs_f64();
    let mut samples = Vec::with_capacity(paths.len());
    for (path, result) in paths.iter().zip(&results) {
        let (seconds, keypoints, width, height) =
            result.lock().expect("result lock").expect("worker result");
        samples.push(seconds);
        if !jsonl {
            println!(
                "{} {}x{}: gpu {:.3}s {} kps",
                path.file_name().unwrap().to_string_lossy(),
                width,
                height,
                seconds,
                keypoints
            );
        }
    }
    samples.sort_by(f64::total_cmp);
    let total: f64 = samples.iter().sum();
    let throughput = paths.len() as f64 / wall_s;
    if jsonl {
        println!(
            "{{\"record\":\"metadata\",\"adapter\":{},\"workers\":{},\"gpu_timestamps_enabled\":false,\"cpu_enabled\":false,\"max_keypoints\":{},\"images\":{}}}",
            json_string(&adapter_name), workers, max_keypoints, paths.len()
        );
        println!(
            "{{\"record\":\"summary\",\"images\":{},\"workers\":{},\"wall_s\":{:.6},\"gpu_sum_image_s\":{:.6},\"gpu_median_s\":{},\"gpu_p95_s\":{},\"throughput_images_per_s\":{:.6}}}",
            paths.len(), workers, wall_s, total,
            emit_json_number(percentile(&samples, 0.5)),
            emit_json_number(percentile(&samples, 0.95)),
            throughput
        );
    } else {
        println!("workers={workers}: wall {wall_s:.3}s, summed image GPU time {total:.3}s, throughput {throughput:.3} images/s");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--images").expect("--images <dir>"));
    let max_images: usize = arg(&args, "--max-images").map_or(5, |v| v.parse().unwrap());
    let workers: usize = arg(&args, "--workers").map_or(1, |v| v.parse().unwrap());
    let run_cpu = !args.iter().any(|a| a == "--no-cpu");
    let jsonl = args.iter().any(|a| a == "--jsonl");
    let gpu_timestamps = args.iter().any(|a| a == "--gpu-timestamps");
    let config = SiftConfig {
        // Match the production photo pipeline's practical feature budget.
        // The library default remains unlimited; pass usize::MAX to benchmark it.
        max_keypoints: arg(&args, "--max-keypoints")
            .map_or(DEFAULT_MAX_KEYPOINTS, |v| v.parse().unwrap()),
        ..SiftConfig::default()
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| {
                let e = e.to_string_lossy().to_ascii_lowercase();
                e == "png" || e == "jpg" || e == "jpeg"
            })
        })
        .collect();
    paths.sort();
    paths.truncate(max_images);

    if workers > 1 {
        assert!(!run_cpu, "--workers > 1 currently requires --no-cpu");
        run_worker_bench(&paths, config.max_keypoints, workers, jsonl);
        return;
    }

    let t = Instant::now();
    let mut gpu = SiftGpu::with_gpu_timestamps(GpuContext::new().expect("gpu"), gpu_timestamps);
    let adapter = &gpu.context().adapter_info;
    let init_s = t.elapsed().as_secs_f64();
    let timestamps_available = gpu
        .context()
        .features
        .contains(wgpu::Features::TIMESTAMP_QUERY);
    if jsonl {
        println!(
            "{{\"record\":\"metadata\",\"adapter\":{},\"backend\":{},\"device_type\":{},\"gpu_init_s\":{:.6},\"max_keypoints\":{},\"max_images\":{},\"cpu_enabled\":{},\"gpu_timestamp_queries_available\":{},\"gpu_timestamps_enabled\":{}}}",
            json_string(&adapter.name),
            json_string(&format!("{:?}", adapter.backend)),
            json_string(&format!("{:?}", adapter.device_type)),
            init_s,
            config.max_keypoints,
            paths.len(),
            run_cpu,
            timestamps_available,
            gpu_timestamps && timestamps_available,
        );
    } else {
        eprintln!(
            "gpu: {} ({:?}, {:?}, init {:.2}s, max keypoints {})",
            adapter.name, adapter.backend, adapter.device_type, init_s, config.max_keypoints
        );
    }
    let (mut cpu_total, mut gpu_total) = (0.0f64, 0.0f64);
    let mut all_dots: Vec<f32> = Vec::new();
    let mut gpu_samples = Vec::with_capacity(paths.len());
    let mut cpu_samples = Vec::with_capacity(paths.len());
    let mut scale_gpu_samples = Vec::with_capacity(paths.len());
    let mut extrema_gpu_samples = Vec::with_capacity(paths.len());
    let mut orientation_gpu_samples = Vec::with_capacity(paths.len());
    let mut describe_gpu_samples = Vec::with_capacity(paths.len());
    for path in &paths {
        let gray = image::open(path).unwrap().to_luma8();
        let (w, h) = (gray.width() as usize, gray.height() as usize);
        let pixels: Vec<f32> = gray.as_raw().iter().map(|&b| b as f32).collect();
        let img = GrayImage::new(w, h, &pixels).unwrap();

        // Warm-up once so pipeline/driver setup is not timed.
        if gpu_total == 0.0 {
            let _ = gpu.extract(&img, &config).unwrap();
        }
        let t = Instant::now();
        let (gk, gd) = gpu.extract(&img, &config).unwrap();
        let tg = t.elapsed().as_secs_f64();
        gpu_total += tg;
        gpu_samples.push(tg);
        let stage_times = gpu
            .take_gpu_timings(!gk.is_empty())
            .filter(|_| gpu_timestamps);
        if let Some(timings) = stage_times {
            scale_gpu_samples.push(timings.scale_space_ms);
            extrema_gpu_samples.push(timings.extrema_ms);
            orientation_gpu_samples.push(timings.orientation_ms);
            if let Some(ms) = timings.descriptor_ms {
                describe_gpu_samples.push(ms);
            }
        }
        let name = path.file_name().unwrap().to_string_lossy();
        if !run_cpu {
            if jsonl {
                println!(
                    "{{\"record\":\"image\",\"file\":{},\"width\":{},\"height\":{},\"gpu_s\":{:.6},\"gpu_keypoints\":{},\"gpu_scale_space_ms\":{},\"gpu_extrema_ms\":{},\"gpu_orientation_ms\":{},\"gpu_descriptor_ms\":{},\"cpu_s\":null,\"cpu_keypoints\":null,\"matched_percent\":null,\"descriptor_dot_p01\":null,\"descriptor_dot_p50\":null}}",
                    json_string(&name), w, h, tg, gk.len(),
                    stage_times.map_or_else(|| "null".to_owned(), |t| format!("{:.6}", t.scale_space_ms)),
                    stage_times.map_or_else(|| "null".to_owned(), |t| format!("{:.6}", t.extrema_ms)),
                    stage_times.map_or_else(|| "null".to_owned(), |t| format!("{:.6}", t.orientation_ms)),
                    stage_times.and_then(|t| t.descriptor_ms).map_or_else(|| "null".to_owned(), |v| format!("{v:.6}")),
                );
            } else {
                if let Some(timings) = stage_times {
                    println!(
                        "{name} {w}x{h}: gpu {tg:.3}s {} kps | scale-space {:.2}ms extrema {:.2}ms orientation {:.2}ms describe {:?}ms",
                        gk.len(),
                        timings.scale_space_ms,
                        timings.extrema_ms,
                        timings.orientation_ms,
                        timings.descriptor_ms
                    );
                } else {
                    println!("{name} {w}x{h}: gpu {tg:.3}s {} kps", gk.len());
                }
            }
            continue;
        }
        let t = Instant::now();
        let (ck, cd) = extract_sift(&img, &config).unwrap();
        let tc = t.elapsed().as_secs_f64();
        cpu_total += tc;
        cpu_samples.push(tc);

        let mut index: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, k) in gk.iter().enumerate() {
            index.entry(key(k)).or_default().push(i);
        }
        let mut matched = 0usize;
        let mut dots: Vec<f32> = Vec::new();
        for (ci, k) in ck.iter().enumerate() {
            let Some(cands) = index.get(&key(k)) else {
                continue;
            };
            let best = cands
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    angle_diff(gk[a].orientation, k.orientation)
                        .total_cmp(&angle_diff(gk[b].orientation, k.orientation))
                })
                .unwrap();
            if angle_diff(gk[best].orientation, k.orientation) < 0.02 {
                matched += 1;
                let dot: f32 = cd[ci].iter().zip(&gd[best]).map(|(a, b)| a * b).sum();
                dots.push(dot);
            }
        }
        dots.sort_by(|a, b| a.total_cmp(b));
        let pct = |q: f64| dots.get(((dots.len() as f64 - 1.0) * q) as usize).copied();
        if jsonl {
            println!(
                "{{\"record\":\"image\",\"file\":{},\"width\":{},\"height\":{},\"gpu_s\":{:.6},\"gpu_keypoints\":{},\"gpu_scale_space_ms\":{},\"gpu_extrema_ms\":{},\"gpu_orientation_ms\":{},\"gpu_descriptor_ms\":{},\"cpu_s\":{:.6},\"cpu_keypoints\":{},\"matched_percent\":{:.6},\"descriptor_dot_p01\":{},\"descriptor_dot_p50\":{}}}",
                json_string(&name), w, h, tg, gk.len(),
                stage_times.map_or_else(|| "null".to_owned(), |t| format!("{:.6}", t.scale_space_ms)),
                stage_times.map_or_else(|| "null".to_owned(), |t| format!("{:.6}", t.extrema_ms)),
                stage_times.map_or_else(|| "null".to_owned(), |t| format!("{:.6}", t.orientation_ms)),
                stage_times.and_then(|t| t.descriptor_ms).map_or_else(|| "null".to_owned(), |v| format!("{v:.6}")),
                tc, ck.len(),
                100.0 * matched as f64 / ck.len().max(1) as f64,
                pct(0.01).map_or_else(|| "null".to_owned(), |v| format!("{v:.8}")),
                pct(0.5).map_or_else(|| "null".to_owned(), |v| format!("{v:.8}")),
            );
        } else {
            println!(
                "{name} {w}x{h}: cpu {tc:.2}s {} kps | gpu {tg:.3}s {} kps ({:.0}x) | cpu kps matched {:.2}% | desc dot p1 {:?} p50 {:?}",
                ck.len(), gk.len(), tc / tg,
                100.0 * matched as f64 / ck.len().max(1) as f64,
                pct(0.01), pct(0.5),
            );
        }
        all_dots.extend(dots);
    }
    if run_cpu {
        all_dots.sort_by(|a, b| a.total_cmp(b));
        let pct = |q: f64| {
            all_dots
                .get(((all_dots.len() as f64 - 1.0) * q) as usize)
                .copied()
        };
        if jsonl {
            gpu_samples.sort_by(f64::total_cmp);
            cpu_samples.sort_by(f64::total_cmp);
            scale_gpu_samples.sort_by(f64::total_cmp);
            extrema_gpu_samples.sort_by(f64::total_cmp);
            orientation_gpu_samples.sort_by(f64::total_cmp);
            describe_gpu_samples.sort_by(f64::total_cmp);
            println!(
                "{{\"record\":\"summary\",\"images\":{},\"gpu_total_s\":{:.6},\"gpu_mean_s\":{:.6},\"gpu_median_s\":{},\"gpu_p95_s\":{},\"gpu_images_per_s\":{:.6},\"gpu_pool_bytes\":{},\"gpu_scale_space_median_ms\":{},\"gpu_scale_space_p95_ms\":{},\"gpu_extrema_median_ms\":{},\"gpu_extrema_p95_ms\":{},\"gpu_orientation_median_ms\":{},\"gpu_orientation_p95_ms\":{},\"gpu_descriptor_median_ms\":{},\"gpu_descriptor_p95_ms\":{},\"cpu_total_s\":{:.6},\"cpu_mean_s\":{:.6},\"cpu_median_s\":{},\"cpu_p95_s\":{},\"cpu_images_per_s\":{:.6},\"descriptor_dot_p001\":{},\"descriptor_dot_p01\":{},\"descriptor_dot_p50\":{}}}",
                paths.len(), gpu_total, gpu_total / paths.len().max(1) as f64,
                emit_json_number(percentile(&gpu_samples, 0.5)),
                emit_json_number(percentile(&gpu_samples, 0.95)),
                paths.len() as f64 / gpu_total,
                gpu.pooled_buffer_bytes(),
                emit_json_number(percentile(&scale_gpu_samples, 0.5)),
                emit_json_number(percentile(&scale_gpu_samples, 0.95)),
                emit_json_number(percentile(&extrema_gpu_samples, 0.5)),
                emit_json_number(percentile(&extrema_gpu_samples, 0.95)),
                emit_json_number(percentile(&orientation_gpu_samples, 0.5)),
                emit_json_number(percentile(&orientation_gpu_samples, 0.95)),
                emit_json_number(percentile(&describe_gpu_samples, 0.5)),
                emit_json_number(percentile(&describe_gpu_samples, 0.95)),
                cpu_total, cpu_total / paths.len().max(1) as f64,
                emit_json_number(percentile(&cpu_samples, 0.5)),
                emit_json_number(percentile(&cpu_samples, 0.95)),
                paths.len() as f64 / cpu_total,
                pct(0.001).map_or_else(|| "null".to_owned(), |v| format!("{v:.8}")),
                pct(0.01).map_or_else(|| "null".to_owned(), |v| format!("{v:.8}")),
                pct(0.5).map_or_else(|| "null".to_owned(), |v| format!("{v:.8}")),
            );
        } else {
            println!(
                "total: cpu {cpu_total:.2}s gpu {gpu_total:.3}s ({:.0}x) | desc dot p0.1 {:?} p1 {:?} p50 {:?}",
                cpu_total / gpu_total, pct(0.001), pct(0.01), pct(0.5)
            );
        }
    } else {
        if jsonl {
            gpu_samples.sort_by(f64::total_cmp);
            scale_gpu_samples.sort_by(f64::total_cmp);
            extrema_gpu_samples.sort_by(f64::total_cmp);
            orientation_gpu_samples.sort_by(f64::total_cmp);
            describe_gpu_samples.sort_by(f64::total_cmp);
            println!(
                "{{\"record\":\"summary\",\"images\":{},\"gpu_total_s\":{:.6},\"gpu_mean_s\":{:.6},\"gpu_median_s\":{},\"gpu_p95_s\":{},\"gpu_images_per_s\":{:.6},\"gpu_pool_bytes\":{},\"gpu_scale_space_median_ms\":{},\"gpu_scale_space_p95_ms\":{},\"gpu_extrema_median_ms\":{},\"gpu_extrema_p95_ms\":{},\"gpu_orientation_median_ms\":{},\"gpu_orientation_p95_ms\":{},\"gpu_descriptor_median_ms\":{},\"gpu_descriptor_p95_ms\":{}}}",
                paths.len(), gpu_total, gpu_total / paths.len().max(1) as f64,
                emit_json_number(percentile(&gpu_samples, 0.5)),
                emit_json_number(percentile(&gpu_samples, 0.95)),
                paths.len() as f64 / gpu_total,
                gpu.pooled_buffer_bytes(),
                emit_json_number(percentile(&scale_gpu_samples, 0.5)),
                emit_json_number(percentile(&scale_gpu_samples, 0.95)),
                emit_json_number(percentile(&extrema_gpu_samples, 0.5)),
                emit_json_number(percentile(&extrema_gpu_samples, 0.95)),
                emit_json_number(percentile(&orientation_gpu_samples, 0.5)),
                emit_json_number(percentile(&orientation_gpu_samples, 0.95)),
                emit_json_number(percentile(&describe_gpu_samples, 0.5)),
                emit_json_number(percentile(&describe_gpu_samples, 0.95)),
            );
        } else {
            println!("total: gpu {gpu_total:.3}s");
        }
    }
}
