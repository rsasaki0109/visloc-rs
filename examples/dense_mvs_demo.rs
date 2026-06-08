//! Dense multi-view stereo reconstruction from a visloc SfM model.
//!
//! The SfM pillar (`--sfm-colmap-out`) produces a *sparse* COLMAP model: refined
//! poses + feature landmarks. This demo densifies it. For each keyframe it runs
//! rectified dense block-matching stereo
//! ([`visloc_rs::vision::dense_stereo`]) to get a per-pixel disparity map,
//! back-projects every valid pixel to a metric 3D point, transforms it into the
//! world frame with the keyframe's refined pose, and fuses the per-frame clouds
//! into one voxel-downsampled dense colored point cloud written as a PLY.
//!
//! Input poses + intrinsics come from the SfM COLMAP model itself (image id N
//! maps to the rectified frame `{N:06}.png`); the stereo baseline is supplied
//! with `--baseline` (EuRoC rectified ≈ 0.110, KITTI ≈ 0.537).
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features image-io --example dense_mvs_demo -- \
//!     --colmap    /tmp/v203_sfm_colmap \
//!     --left-dir  /tmp/V2_03_rect/image_0 \
//!     --right-dir /tmp/V2_03_rect/image_1 \
//!     --baseline  0.110 \
//!     --stride 1 --out-ply /tmp/v203_dense.ply
//! ```

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "this example requires the `image-io` feature; rebuild with \
         `cargo run --release --features image-io --example dense_mvs_demo`"
    );
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::run()
}

#[cfg(feature = "image-io")]
mod imp {
    use std::collections::HashMap;
    use std::env;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;

    use visloc_rs::io::colmap::read_colmap_text_model;
    use visloc_rs::io::images::read_common_image;
    use visloc_rs::vision::dense_stereo::{dense_stereo_points, DenseStereoConfig};

    /// Per-voxel accumulator: `(sum_world_xyz, sum_intensity, point_count)`.
    type VoxelAcc = ([f64; 3], f64, u32);

    struct Args {
        colmap: PathBuf,
        left_dir: PathBuf,
        right_dir: PathBuf,
        baseline: f64,
        stride: usize,
        max_frames: usize,
        out_ply: PathBuf,
        min_depth: f64,
        max_depth: f64,
        voxel: f64,
        min_voxel_count: u32,
        cfg: DenseStereoConfig,
    }

    fn parse_args() -> Result<Args, String> {
        let mut colmap = None;
        let mut left_dir = None;
        let mut right_dir = None;
        let mut baseline = None;
        let mut stride = 1usize;
        let mut max_frames = usize::MAX;
        let mut out_ply = PathBuf::from("dense.ply");
        let mut min_depth = 0.2;
        let mut max_depth = 20.0;
        let mut voxel = 0.02;
        let mut min_voxel_count = 1u32;
        let mut cfg = DenseStereoConfig::default();
        let mut a: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < a.len() {
            match a[i].as_str() {
                "--colmap" => colmap = Some(PathBuf::from(a.remove(i + 1))),
                "--left-dir" => left_dir = Some(PathBuf::from(a.remove(i + 1))),
                "--right-dir" => right_dir = Some(PathBuf::from(a.remove(i + 1))),
                "--baseline" => {
                    baseline = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
                }
                "--stride" => stride = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
                "--max-frames" => {
                    max_frames = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
                }
                "--out-ply" => out_ply = PathBuf::from(a.remove(i + 1)),
                "--min-depth" => min_depth = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
                "--max-depth" => max_depth = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
                "--voxel" => voxel = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
                "--min-voxel-count" => {
                    min_voxel_count = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
                }
                "--max-disparity" => {
                    cfg.max_disparity = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
                }
                "--block-radius" => {
                    cfg.block_radius = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
                }
                "--max-block-diff" => {
                    cfg.max_block_diff = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
                }
                other => return Err(format!("unknown argument: {other}")),
            }
            i += 1;
        }
        Ok(Args {
            colmap: colmap.ok_or("--colmap is required")?,
            left_dir: left_dir.ok_or("--left-dir is required")?,
            right_dir: right_dir.ok_or("--right-dir is required")?,
            baseline: baseline.ok_or("--baseline is required")?,
            stride,
            max_frames,
            out_ply,
            min_depth,
            max_depth,
            voxel,
            min_voxel_count,
            cfg,
        })
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args()?;
        let map = read_colmap_text_model(&args.colmap)?;
        let camera = map
            .cameras
            .values()
            .next()
            .ok_or("colmap model has no camera")?
            .clone();
        println!(
            "loaded SfM model: {} keyframes, {} landmarks, camera {}x{}",
            map.keyframes.len(),
            map.landmarks.len(),
            camera.width,
            camera.height,
        );

        // Keyframes sorted by id; the rectified frame for image id N is {N:06}.png.
        let mut ids: Vec<u64> = map.keyframes.keys().copied().collect();
        ids.sort_unstable();

        // Voxel hash → (sum_world, sum_intensity, count) for downsampling.
        let mut voxels: HashMap<(i64, i64, i64), VoxelAcc> = HashMap::new();
        let inv_voxel = 1.0 / args.voxel;
        let mut frames_used = 0usize;
        let mut raw_points = 0usize;

        for (k, &id) in ids.iter().enumerate() {
            if frames_used >= args.max_frames {
                break;
            }
            if k % args.stride != 0 {
                continue;
            }
            let Some(pose) = map.keyframes[&id].frame.pose.clone() else {
                continue;
            };
            let name = format!("{id:06}.png");
            let left = match read_common_image(args.left_dir.join(&name)) {
                Ok(im) => im,
                Err(_) => continue,
            };
            let right = match read_common_image(args.right_dir.join(&name)) {
                Ok(im) => im,
                Err(_) => continue,
            };
            let pts = dense_stereo_points(&left, &right, &camera, args.baseline, &args.cfg);
            let cam_to_world = pose.camera_to_world();
            for p in &pts {
                if p.point_cam.z < args.min_depth || p.point_cam.z > args.max_depth {
                    continue;
                }
                let w = cam_to_world.transform_point(&p.point_cam);
                raw_points += 1;
                let key = (
                    (w.x * inv_voxel).floor() as i64,
                    (w.y * inv_voxel).floor() as i64,
                    (w.z * inv_voxel).floor() as i64,
                );
                let e = voxels.entry(key).or_insert(([0.0; 3], 0.0, 0));
                e.0[0] += w.x;
                e.0[1] += w.y;
                e.0[2] += w.z;
                e.1 += p.intensity as f64;
                e.2 += 1;
            }
            frames_used += 1;
            if frames_used % 10 == 0 {
                println!(
                    "  fused {frames_used} frames, {} voxels, {raw_points} raw points",
                    voxels.len()
                );
            }
        }

        // Multi-view consistency filter: a real surface voxel is hit by many
        // rays across frames; an isolated bad-disparity outlier is hit by few.
        // Dropping voxels below `min_voxel_count` removes most depth noise.
        let kept: Vec<&VoxelAcc> = voxels
            .values()
            .filter(|(_, _, count)| *count >= args.min_voxel_count)
            .collect();

        // Write the fused cloud as an ASCII PLY (x y z r g b).
        let file = File::create(&args.out_ply)?;
        let mut w = BufWriter::new(file);
        writeln!(w, "ply")?;
        writeln!(w, "format ascii 1.0")?;
        writeln!(w, "element vertex {}", kept.len())?;
        writeln!(w, "property float x")?;
        writeln!(w, "property float y")?;
        writeln!(w, "property float z")?;
        writeln!(w, "property uchar red")?;
        writeln!(w, "property uchar green")?;
        writeln!(w, "property uchar blue")?;
        writeln!(w, "end_header")?;
        for (sum, intensity, count) in &kept {
            let n = *count as f64;
            let g = ((intensity / n) * 255.0).round().clamp(0.0, 255.0) as u8;
            writeln!(
                w,
                "{} {} {} {} {} {}",
                sum[0] / n,
                sum[1] / n,
                sum[2] / n,
                g,
                g,
                g
            )?;
        }
        w.flush()?;

        println!(
            "dense MVS: {frames_used} frames -> {raw_points} raw points -> {} voxels ({} m) \
             -> {} kept (count>={}) -> {}",
            voxels.len(),
            args.voxel,
            kept.len(),
            args.min_voxel_count,
            args.out_ply.display(),
        );
        Ok(())
    }
}
