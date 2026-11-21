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
//! (`k1 k2 p1 p2`) or `FULL_OPENCV` with its rational `k4..k6` at 0, so the
//! image can carry a real lens's distortion.
//!
//! Each request is one line of 12 numbers, the world-to-camera rotation
//! (row-major) and translation of the splat's frame, in COLMAP's convention
//! (`x_cam = R x_world + t`, camera x right, y down, z forward). The reply is
//! `width * height * 3` bytes of 8-bit RGB, row by row, with nothing else;
//! with `--depth`, followed by `width * height` little-endian f32 median
//! depths (camera z in the splat's units, 0 where nothing is hit).
//! Diagnostics go to stderr. The process ends at the end of stdin.
//! `ros2/visloc-ros2/gs/gs_sim.py` drives it.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use nalgebra::{Matrix3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::lens::{Lens, Projection};
use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_render::{GpuContext, Renderer};

fn parse_camera(spec: &str) -> Result<PinholeCamera, String> {
    let parts: Vec<&str> = spec.split_whitespace().collect();
    let model = parts.first().copied().unwrap_or("");
    let want = match model {
        "PINHOLE" => 7,
        "OPENCV" => 11,
        "FULL_OPENCV" => 15,
        _ => 0,
    };
    if want == 0 || parts.len() != want {
        return Err(format!(
            "--camera wants \"PINHOLE W H FX FY CX CY\", \"OPENCV ... K1 K2 P1 P2\" or \
             \"FULL_OPENCV ... K1 K2 P1 P2 K3 K4 K5 K6\", got {spec:?}"
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
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ply" => ply = args.next().map(PathBuf::from),
            "--camera" => camera = args.next(),
            "--sh-degree" => sh_degree = args.next().and_then(|v| v.parse().ok()),
            "--depth" => with_depth = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <path> is required")?;
    let camera = parse_camera(
        &camera
            .ok_or("--camera \"PINHOLE W H FX FY CX CY\" (or OPENCV, FULL_OPENCV) is required")?,
    )?;
    let scene = load_ply(&ply)?;
    let mut renderer = Renderer::new(GpuContext::new()?, &scene, camera.width, camera.height)?;
    if let Some(d) = sh_degree {
        renderer.set_active_sh_degree(Some(d));
    }
    eprintln!(
        "gsplat_render_server: {} gaussians, {}x{}",
        scene.len(),
        camera.width,
        camera.height
    );

    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut bytes = Vec::with_capacity((camera.width * camera.height * 3) as usize);
    for line in stdin.lock().lines() {
        let line = line?;
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
