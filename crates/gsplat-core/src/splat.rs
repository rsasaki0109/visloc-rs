//! Reader/writer for the antimatter15 `.splat` format.
//!
//! The format is a headerless array of 32-byte records, one per Gaussian:
//!
//! | offset | bytes | field | encoding |
//! | --- | --- | --- | --- |
//! | 0 | 12 | position `(x, y, z)` | 3 × f32 little-endian |
//! | 12 | 12 | scale `(sx, sy, sz)` | 3 × f32 little-endian, **linear** |
//! | 24 | 4 | colour RGBA | 4 × u8; RGB linear colour × 255, A opacity × 255 |
//! | 28 | 4 | rotation `(w, x, y, z)` | 4 × u8; unit quat × 128 + 128 |
//!
//! `.splat` has no spherical harmonics and stores *linear* scale and colour,
//! so it cannot round-trip the training representation exactly: [`load_splat`]
//! reconstructs a degree-0 [`Scene`] with `scale_log = ln(scale)` and
//! `sh_dc = (rgb - 0.5) / SH_C0` so the renderer's SH path reproduces the stored
//! colour, and [`save_splat`] clamps and quantizes the inverse. This is
//! intentional: `.splat` is the lightweight viewer/delivery format, while the
//! Inria `.ply` (see [`crate::ply`]) is lossless.

use std::fs;
use std::io;
use std::path::Path;

use nalgebra::{Quaternion, Vector3};

use crate::gaussian::Scene;
use crate::sh::SH_C0;

/// One decoded `.splat` record in its on-disk (linear) form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplatRecord {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub color: [u8; 3],
    pub opacity: u8,
    pub rotation: [u8; 4],
}

impl SplatRecord {
    const BYTES: usize = 32;
}

/// Errors from `.splat` parsing.
#[derive(Debug, thiserror::Error)]
pub enum SplatError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error(".splat file length {len} is not a multiple of 32 bytes")]
    BadLength { len: usize },
    #[error("no finite gaussians to write")]
    Empty,
}

/// Decode a `.splat` blob into records.
pub fn parse_splat(bytes: &[u8]) -> Result<Vec<SplatRecord>, SplatError> {
    if !bytes.len().is_multiple_of(SplatRecord::BYTES) {
        return Err(SplatError::BadLength { len: bytes.len() });
    }
    let mut out = Vec::with_capacity(bytes.len() / SplatRecord::BYTES);
    for chunk in bytes.chunks_exact(SplatRecord::BYTES) {
        let position = [
            f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
            f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
        ];
        let scale = [
            f32::from_le_bytes(chunk[12..16].try_into().unwrap()),
            f32::from_le_bytes(chunk[16..20].try_into().unwrap()),
            f32::from_le_bytes(chunk[20..24].try_into().unwrap()),
        ];
        let color = [chunk[24], chunk[25], chunk[26]];
        let opacity = chunk[27];
        let rotation = [chunk[28], chunk[29], chunk[30], chunk[31]];
        out.push(SplatRecord {
            position,
            scale,
            color,
            opacity,
            rotation,
        });
    }
    Ok(out)
}

/// Read a `.splat` file into a degree-0 [`Scene`].
pub fn load_splat(path: impl AsRef<Path>) -> Result<Scene, SplatError> {
    let bytes = fs::read(path)?;
    scene_from_splat(&bytes)
}

/// Build a degree-0 [`Scene`] from a `.splat` blob.
pub fn scene_from_splat(bytes: &[u8]) -> Result<Scene, SplatError> {
    let records = parse_splat(bytes)?;
    let mut gaussians = Vec::with_capacity(records.len());
    for r in records {
        // Guard against zero/negative linear scale before taking ln.
        let lin = Vector3::new(
            r.scale[0].max(f32::MIN_POSITIVE),
            r.scale[1].max(f32::MIN_POSITIVE),
            r.scale[2].max(f32::MIN_POSITIVE),
        );
        let scale_log = lin.map(f32::ln);
        // Rotation bytes are `unit_quat * 128 + 128` in `(w, x, y, z)` order.
        let qw = r.rotation[0] as f32 / 128.0 - 1.0;
        let qx = r.rotation[1] as f32 / 128.0 - 1.0;
        let qy = r.rotation[2] as f32 / 128.0 - 1.0;
        let qz = r.rotation[3] as f32 / 128.0 - 1.0;
        let rotation = Quaternion::new(qw, qx, qy, qz);
        // Invert the renderer's `sh_dc * SH_C0 + 0.5` so the linear colour is
        // reproduced through the (degree-0) SH path.
        let sh_dc = [
            (r.color[0] as f32 / 255.0 - 0.5) / SH_C0,
            (r.color[1] as f32 / 255.0 - 0.5) / SH_C0,
            (r.color[2] as f32 / 255.0 - 0.5) / SH_C0,
        ];
        let opacity_linear = r.opacity as f32 / 255.0;
        let opacity_logit = logit(opacity_linear);
        gaussians.push(crate::gaussian::Gaussian {
            mean: Vector3::new(r.position[0], r.position[1], r.position[2]),
            scale_log,
            rotation,
            opacity_logit,
            sh_dc,
            sh_rest: Vec::new(),
            sh_degree: 0,
        });
    }
    Ok(Scene::new(gaussians, 0))
}

/// Encode a [`Scene`] to the 32-byte `.splat` layout.
///
/// SH and log-scale information are lost (see the module docs); the scene is
/// assumed to be degree 0 for colour purposes and any `sh_rest` is ignored with
/// the DC term used as the flat colour.
pub fn encode_splat(scene: &Scene) -> Result<Vec<u8>, SplatError> {
    if scene.is_empty() {
        return Err(SplatError::Empty);
    }
    let mut out = Vec::with_capacity(scene.len() * SplatRecord::BYTES);
    for g in &scene.gaussians {
        let scale = g.scale();
        let q = g.unit_rotation();
        // Degree-0 colour through the same path the renderer uses.
        let color = [
            (g.sh_dc[0] * SH_C0 + 0.5).clamp(0.0, 1.0),
            (g.sh_dc[1] * SH_C0 + 0.5).clamp(0.0, 1.0),
            (g.sh_dc[2] * SH_C0 + 0.5).clamp(0.0, 1.0),
        ];
        out.extend_from_slice(&g.mean.x.to_le_bytes());
        out.extend_from_slice(&g.mean.y.to_le_bytes());
        out.extend_from_slice(&g.mean.z.to_le_bytes());
        out.extend_from_slice(&scale.x.to_le_bytes());
        out.extend_from_slice(&scale.y.to_le_bytes());
        out.extend_from_slice(&scale.z.to_le_bytes());
        out.push((color[0] * 255.0).round() as u8);
        out.push((color[1] * 255.0).round() as u8);
        out.push((color[2] * 255.0).round() as u8);
        out.push((g.opacity() * 255.0).round() as u8);
        for c in [q.w, q.i, q.j, q.k] {
            out.push(((c * 128.0 + 128.0).clamp(0.0, 255.0)).round() as u8);
        }
    }
    Ok(out)
}

/// Write a [`Scene`] as a `.splat` file.
pub fn save_splat(scene: &Scene, path: impl AsRef<Path>) -> Result<(), SplatError> {
    fs::write(path, encode_splat(scene)?)?;
    Ok(())
}

/// The indices of the `max` most important gaussians, most important first:
/// opacity times volume, the order antimatter15's viewer sorts by. Ties keep
/// the scene's order, so two scenes of the same geometry give the same
/// order.
pub fn importance_order(scene: &Scene, max: usize) -> Vec<usize> {
    let importance = |g: &crate::gaussian::Gaussian| {
        g.opacity() * (g.scale_log.x + g.scale_log.y + g.scale_log.z).exp()
    };
    let mut order: Vec<usize> = (0..scene.len()).collect();
    order.sort_by(|&a, &b| {
        importance(&scene.gaussians[b]).total_cmp(&importance(&scene.gaussians[a]))
    });
    order.truncate(max);
    order
}

#[inline]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::Gaussian;

    fn sample_scene() -> Scene {
        Scene::new(
            vec![Gaussian {
                mean: Vector3::new(1.0, -2.0, 3.0),
                scale_log: Vector3::new(0.0, 0.0, 0.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                opacity_logit: 0.0,
                sh_dc: [0.0, 0.0, 0.0],
                sh_rest: Vec::new(),
                sh_degree: 0,
            }],
            0,
        )
    }

    #[test]
    fn record_size_is_32() {
        let bytes = encode_splat(&sample_scene()).unwrap();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn position_round_trips() {
        let bytes = encode_splat(&sample_scene()).unwrap();
        let scene = scene_from_splat(&bytes).unwrap();
        let m = scene.gaussians[0].mean;
        assert!((m.x - 1.0).abs() < 1e-6);
        assert!((m.y + 2.0).abs() < 1e-6);
        assert!((m.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn scale_round_trips_through_log() {
        let bytes = encode_splat(&sample_scene()).unwrap();
        let scene = scene_from_splat(&bytes).unwrap();
        let s = scene.gaussians[0].scale();
        assert!((s.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn opacity_round_trips_approximate() {
        let bytes = encode_splat(&sample_scene()).unwrap();
        let scene = scene_from_splat(&bytes).unwrap();
        // 0.5 encodes to 128, 128/255 re-sigmoids to ~0.502.
        assert!((scene.gaussians[0].opacity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn rejects_bad_length() {
        assert!(matches!(
            parse_splat(&[0u8; 33]),
            Err(SplatError::BadLength { .. })
        ));
    }

    #[test]
    fn rejects_empty_scene() {
        assert!(matches!(
            encode_splat(&Scene::new(vec![], 0)),
            Err(SplatError::Empty)
        ));
    }
}
