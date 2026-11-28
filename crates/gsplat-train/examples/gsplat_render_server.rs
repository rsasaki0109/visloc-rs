//! Render a trained splat on request, for a simulator that needs camera
//! images: the splat is loaded once, then each line on stdin is a camera pose
//! and the answer on stdout is the raw image.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features gpu --example gsplat_render_server -- \
//!     --ply scene.ply --camera "PINHOLE 640 480 520.9 521.0 325.1 249.7" [--sh-degree 3]
//! ```
//!
//! `--camera` is a COLMAP camera row without its id: `PINHOLE`, `OPENCV`
//! (`k1 k2 p1 p2`), `FULL_OPENCV` with its rational `k4..k6` at 0 or
//! `OPENCV_FISHEYE` (`k1 k2 k3 k4`, equidistant), so the image can carry a
//! real lens's distortion and a model trained through a fisheye can be
//! rendered through the same projection.
//!
//! Each request is one line of 12 numbers, the world-to-camera rotation
//! (row-major) and translation of the splat's frame, in COLMAP's convention
//! (`x_cam = R x_world + t`, camera x right, y down, z forward). The reply is
//! `width * height * 3` bytes of 8-bit RGB, row by row, with nothing else;
//! with `--depth`, followed by `width * height` little-endian f32 median
//! depths (camera z in the splat's units, 0 where nothing is hit; for
//! `OPENCV_FISHEYE` the distance from the camera centre, as the trainer's
//! depth channel).
//! Diagnostics go to stderr. The process ends at the end of stdin.
//!
//! Moving objects (people, bicycles) can be splats too: `--dynamic N`
//! keeps `N` extra gaussian slots after the scene, empty at first. A line
//! `dyn K` (K <= N) before a pose request is followed by `K * 14`
//! little-endian f32s, per gaussian `x y z  log_sx log_sy log_sz  qw qx qy qz
//! opacity_logit  r g b` (colour in 0..1, view-independent). They fill the
//! first `K` slots, the rest are emptied, and they stay until the next
//! `dyn` line. So the objects are sorted, blended and occluded with the
//! scene by the same rasterizer, and show in the depth.
//! `ros2/visloc-ros2/gs/gs_sim.py` drives it.
//!
//! `--near M` skips gaussians nearer than `M` m to the camera (default 0.1):
//! a chase camera passing a street sign does not see it fill the frame.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;

use nalgebra::Quaternion;
use nalgebra::{Matrix3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::{sh_rest_coeffs_per_channel, Gaussian};
use visloc_gsplat_core::lens::{Lens, Projection};
use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_core::sh::SH_C0;
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_render::{GpuContext, Renderer};

/// Floats per dynamic gaussian in a `dyn` block.
const DYN_FLOATS: usize = 14;

/// A gaussian from a `dyn` record (empty: transparent and tiny).
fn dynamic_gaussian(v: Option<&[f32]>, sh_degree: u32) -> Gaussian {
    let rest = vec![0.0; 3 * sh_rest_coeffs_per_channel(sh_degree)];
    match v {
        Some(v) => Gaussian {
            mean: Vector3::new(v[0], v[1], v[2]),
            scale_log: Vector3::new(v[3], v[4], v[5]),
            rotation: Quaternion::new(v[6], v[7], v[8], v[9]),
            opacity_logit: v[10],
            sh_dc: [
                (v[11] - 0.5) / SH_C0,
                (v[12] - 0.5) / SH_C0,
                (v[13] - 0.5) / SH_C0,
            ],
            sh_rest: rest,
            sh_degree,
        },
        None => Gaussian {
            mean: Vector3::new(0.0, 0.0, -1.0e4),
            scale_log: Vector3::new(-12.0, -12.0, -12.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: -30.0,
            sh_dc: [0.0; 3],
            sh_rest: rest,
            sh_degree,
        },
    }
}

fn parse_camera(spec: &str) -> Result<PinholeCamera, String> {
    let parts: Vec<&str> = spec.split_whitespace().collect();
    let model = parts.first().copied().unwrap_or("");
    let want = match model {
        "PINHOLE" => 7,
        "OPENCV" => 11,
        "FULL_OPENCV" => 15,
        "OPENCV_FISHEYE" => 11,
        _ => 0,
    };
    if want == 0 || parts.len() != want {
        return Err(format!(
            "--camera wants \"PINHOLE W H FX FY CX CY\", \"OPENCV ... K1 K2 P1 P2\", \
             \"FULL_OPENCV ... K1 K2 P1 P2 K3 K4 K5 K6\" or \
             \"OPENCV_FISHEYE ... K1 K2 K3 K4\", got {spec:?}"
        ));
    }
    let n = |i: usize| -> Result<f32, String> {
        parts[i]
            .parse::<f32>()
            .map_err(|e| format!("--camera field {i}: {e}"))
    };
    let camera = PinholeCamera::new(n(1)? as u32, n(2)? as u32, n(3)?, n(4)?, n(5)?, n(6)?);
    if model == "PINHOLE" {
        return Ok(camera);
    }
    if model == "OPENCV_FISHEYE" {
        // Kannala-Brandt k1..k4, as `colmap_scene::lens_of` reads them.
        let coeffs = [n(7)?, n(8)?, n(9)?, n(10)?, 0.0, 0.0, 0.0, 0.0];
        return Ok(camera.with_lens(Lens::new(Projection::Equidistant, coeffs)));
    }
    // COLMAP's order (k1 k2 p1 p2 [k3 k4 k5 k6]) into the lens's
    // [k1, k2, k3, k4, p1, p2, sx1, sy1]; the rational k4..k6 are not modelled.
    let k3 = if model == "FULL_OPENCV" { n(11)? } else { 0.0 };
    if model == "FULL_OPENCV" && (n(12)? != 0.0 || n(13)? != 0.0 || n(14)? != 0.0) {
        return Err("FULL_OPENCV's K4..K6 (the rational part) must be 0".into());
    }
    let coeffs = [n(7)?, n(8)?, k3, 0.0, n(9)?, n(10)?, 0.0, 0.0];
    Ok(camera.with_lens(Lens::new(Projection::Pinhole, coeffs)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut camera: Option<String> = None;
    let mut sh_degree: Option<u32> = None;
    let mut with_depth = false;
    let mut dynamic = 0usize;
    let mut near = 0.0f32;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ply" => ply = args.next().map(PathBuf::from),
            "--camera" => camera = args.next(),
            "--sh-degree" => sh_degree = args.next().and_then(|v| v.parse().ok()),
            "--depth" => with_depth = true,
            "--near" => {
                near = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--near M: the nearest depth drawn, m")?
            }
            "--dynamic" => {
                dynamic = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--dynamic N: the number of moving-object gaussian slots")?
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <path> is required")?;
    let camera = parse_camera(&camera.ok_or(
        "--camera \"PINHOLE W H FX FY CX CY\" (or OPENCV, FULL_OPENCV, OPENCV_FISHEYE) is required",
    )?)?;
    let mut scene = load_ply(&ply)?;
    let first_dynamic = scene.len();
    let sh_degree_scene = scene.sh_degree;
    scene
        .gaussians
        .extend((0..dynamic).map(|_| dynamic_gaussian(None, sh_degree_scene)));
    let mut renderer = Renderer::new(GpuContext::new()?, &scene, camera.width, camera.height)?;
    renderer.set_near_plane(near);
    if let Some(d) = sh_degree {
        renderer.set_active_sh_degree(Some(d));
    }
    eprintln!(
        "gsplat_render_server: {} gaussians (+{dynamic} dynamic slots), {}x{}",
        first_dynamic, camera.width, camera.height
    );

    let mut stdin = std::io::BufReader::new(std::io::stdin().lock());
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut bytes = Vec::with_capacity((camera.width * camera.height * 3) as usize);
    let mut line = String::new();
    loop {
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        if let Some(count) = line.trim().strip_prefix("dyn") {
            let k: usize = count.trim().parse().map_err(|e| format!("dyn K: {e}"))?;
            if k > dynamic {
                return Err(format!("dyn {k}: only {dynamic} --dynamic slots").into());
            }
            let mut raw = vec![0u8; k * DYN_FLOATS * 4];
            stdin.read_exact(&mut raw)?;
            let floats: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let gaussians = (0..dynamic)
                .map(|i| {
                    dynamic_gaussian(
                        (i < k).then(|| &floats[i * DYN_FLOATS..(i + 1) * DYN_FLOATS]),
                        sh_degree_scene,
                    )
                })
                .collect();
            let packed = PackedScene::from_scene(&visloc_gsplat_core::gaussian::Scene::new(
                gaussians,
                sh_degree_scene,
            ));
            renderer
                .scene
                .write_range(&renderer.ctx, first_dynamic, &packed);
            continue;
        }
        let v: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect::<Result<_, _>>()?;
        if v.len() != 12 {
            return Err(
                format!("a request is 12 numbers (R row-major, t), got {}", v.len()).into(),
            );
        }
        let rotation = Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
        let translation = Vector3::new(v[9], v[10], v[11]);
        let view = CameraView::new(rotation, translation, camera);
        let (image, depth) = if with_depth {
            let (image, depth, _) = renderer.render_depth(&view, [0.0; 3]);
            (image, Some(depth))
        } else {
            (renderer.render(&view, [0.0; 3]), None)
        };
        bytes.clear();
        bytes.extend(
            image
                .rgb
                .iter()
                .flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)),
        );
        if let Some(depth) = depth {
            bytes.extend(depth.iter().flat_map(|d| d.to_le_bytes()));
        }
        out.write_all(&bytes)?;
        out.flush()?;
    }
    Ok(())
}
