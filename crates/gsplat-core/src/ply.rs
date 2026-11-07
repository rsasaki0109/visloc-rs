//! Reader/writer for the Inria 3DGS `.ply` format.
//!
//! A `.ply` file written by `gaussian-splatting` / `gsplat` is a binary
//! little-endian PLY whose single `vertex` element carries, per Gaussian:
//!
//! ```text
//! x, y, z                 f32   world-space mean
//! nx, ny, nz              f32   normals (usually zero; not used)
//! f_dc_0..2               f32   degree-0 SH per RGB channel
//! f_rest_0..(3*K-1)       f32   higher-order SH, channel-major
//! opacity                 f32   opacity logit
//! scale_0..2              f32   log-scale per axis
//! rot_0..3                f32   quaternion (w, x, y, z), unnormalized
//! ```
//!
//! `K = sh_rest_coeffs_per_channel(degree)` (0, 3, 8, 15 for degrees 0..3).
//! Unlike `.splat`, `.ply` stores the raw (pre-activation) values and the full
//! SH, so it is the lossless interop format used for training.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use nalgebra::{Quaternion, Vector3};

use crate::gaussian::{sh_rest_coeffs_per_channel, Gaussian, Scene};

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("not a PLY file (missing magic)")]
    BadMagic,
    #[error("unsupported PLY format (only binary_little_endian is supported)")]
    UnsupportedFormat,
    #[error("missing `vertex` element")]
    MissingVertex,
    #[error("could not find vertex property `{0}`")]
    MissingProperty(String),
    #[error("vertex element declares zero vertices")]
    Empty,
    #[error("header property count {declared} does not match data bytes {actual}")]
    SizeMismatch { declared: usize, actual: usize },
    #[error("no gaussians to write")]
    NothingToWrite,
    #[error("extra property `{name}` has {values} values for {gaussians} gaussians")]
    ExtraLength {
        name: String,
        values: usize,
        gaussians: usize,
    },
}

/// Parse the PLY header, returning the vertex property names in order and the
/// byte offset where the binary body starts.
fn parse_header(bytes: &[u8]) -> Result<(Vec<String>, usize, usize), PlyError> {
    if !bytes.starts_with(b"ply") {
        return Err(PlyError::BadMagic);
    }
    let text_end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .map(|p| p + 11)
        .or_else(|| {
            bytes
                .windows(10)
                .position(|w| w == b"end_header")
                .map(|p| p + 10)
        })
        .ok_or(PlyError::BadMagic)?;
    let header = std::str::from_utf8(&bytes[..text_end]).map_err(|_| PlyError::BadMagic)?;
    let mut properties = Vec::new();
    let mut in_vertex = false;
    let mut vertex_count = None;
    for line in header.lines() {
        let line = line.trim();
        if line.starts_with("format ") {
            if !line.contains("binary_little_endian") {
                return Err(PlyError::UnsupportedFormat);
            }
        } else if let Some(rest) = line.strip_prefix("element ") {
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("");
            let count: usize = parts.next().and_then(|c| c.parse().ok()).unwrap_or(0);
            in_vertex = name == "vertex";
            if in_vertex {
                vertex_count = Some(count);
            }
        } else if in_vertex {
            if let Some(rest) = line.strip_prefix("property ") {
                // `property <type> <name>`; we only support scalar f32 here.
                let mut parts = rest.split_whitespace();
                let _ty = parts.next();
                if let Some(name) = parts.next() {
                    properties.push(name.to_owned());
                }
            }
        }
    }
    let vertex_count = vertex_count.ok_or(PlyError::MissingVertex)?;
    Ok((properties, vertex_count, text_end))
}

/// Read a binary-little-endian Inria `.ply` into a [`Scene`].
pub fn load_ply(path: impl AsRef<Path>) -> Result<Scene, PlyError> {
    let bytes = fs::read(path)?;
    scene_from_ply(&bytes)
}

/// Decode an Inria `.ply` blob into a [`Scene`].
pub fn scene_from_ply(bytes: &[u8]) -> Result<Scene, PlyError> {
    let (properties, vertex_count, body_start) = parse_header(bytes)?;
    let body = &bytes[body_start..];
    let stride = properties.len() * 4;
    if stride == 0 {
        return Err(PlyError::MissingVertex);
    }
    let expected = vertex_count * stride;
    if body.len() < expected {
        return Err(PlyError::SizeMismatch {
            declared: expected,
            actual: body.len(),
        });
    }
    let index: HashMap<&str, usize> = properties
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let req = |name: &str| -> Result<usize, PlyError> {
        index
            .get(name)
            .copied()
            .ok_or_else(|| PlyError::MissingProperty(name.to_owned()))
    };
    let i_x = req("x")?;
    let i_y = req("y")?;
    let i_z = req("z")?;
    // Look every property up by name: exporters order them differently (brush
    // sorts alphabetically), so do not assume f_dc_1 follows f_dc_0.
    let i_dc = [req("f_dc_0")?, req("f_dc_1")?, req("f_dc_2")?];
    let i_opacity = req("opacity")?;
    let i_s = [req("scale_0")?, req("scale_1")?, req("scale_2")?];
    let i_r = [req("rot_0")?, req("rot_1")?, req("rot_2")?, req("rot_3")?];

    // Discover the SH degree from the highest present f_rest index.
    let mut rest_indices: Vec<(usize, usize)> = Vec::new();
    let mut k = 0usize;
    while let Some(&idx) = index.get(format!("f_rest_{k}").as_str()) {
        rest_indices.push((k, idx));
        k += 1;
    }
    let rest_total = rest_indices.len();
    if !rest_total.is_multiple_of(3) {
        return Err(PlyError::MissingProperty("f_rest_*".to_owned()));
    }
    let coeffs_per_channel = rest_total / 3;
    let sh_degree = match coeffs_per_channel {
        0 => 0,
        3 => 1,
        8 => 2,
        15 => 3,
        _ => {
            return Err(PlyError::MissingProperty(format!(
                "unsupported f_rest count {rest_total}"
            )))
        }
    };

    let f32_at = |row: usize, col: usize| -> f32 {
        let off = row * stride + col * 4;
        f32::from_le_bytes(body[off..off + 4].try_into().unwrap())
    };

    let mut gaussians = Vec::with_capacity(vertex_count);
    for row in 0..vertex_count {
        let mean = Vector3::new(f32_at(row, i_x), f32_at(row, i_y), f32_at(row, i_z));
        // f_rest_0..2 are the first higher-order R/G/B coefficients; the DC is
        // in f_dc_0..2. The full rest layout is channel-major once re-ordered.
        let sh_dc = i_dc.map(|i| f32_at(row, i));
        let mut sh_rest = vec![0.0f32; rest_total];
        // PLY stores f_rest channel-major already: [R..., G..., B...].
        for (k, idx) in &rest_indices {
            sh_rest[*k] = f32_at(row, *idx);
        }
        let scale_log = Vector3::new(
            f32_at(row, i_s[0]),
            f32_at(row, i_s[1]),
            f32_at(row, i_s[2]),
        );
        let rotation = Quaternion::new(
            f32_at(row, i_r[0]),
            f32_at(row, i_r[1]),
            f32_at(row, i_r[2]),
            f32_at(row, i_r[3]),
        );
        let opacity_logit = f32_at(row, i_opacity);
        gaussians.push(Gaussian {
            mean,
            scale_log,
            rotation,
            opacity_logit,
            sh_dc,
            sh_rest,
            sh_degree,
        });
    }
    Ok(Scene::new(gaussians, sh_degree))
}

/// The float vertex properties `names` of a `.ply` blob, such as the extra
/// ones [`encode_ply_with`] writes: one value per gaussian for each name the
/// file has, `None` for the others.
pub fn ply_properties(bytes: &[u8], names: &[&str]) -> Result<Vec<Option<Vec<f32>>>, PlyError> {
    let (properties, vertex_count, body_start) = parse_header(bytes)?;
    let body = &bytes[body_start..];
    let stride = properties.len() * 4;
    let expected = vertex_count * stride;
    if body.len() < expected {
        return Err(PlyError::SizeMismatch {
            declared: expected,
            actual: body.len(),
        });
    }
    Ok(names
        .iter()
        .map(|name| {
            let col = properties.iter().position(|p| p == name)?;
            Some(
                (0..vertex_count)
                    .map(|row| {
                        let off = row * stride + col * 4;
                        f32::from_le_bytes(body[off..off + 4].try_into().unwrap())
                    })
                    .collect(),
            )
        })
        .collect())
}

/// Encode a [`Scene`] as a binary-little-endian Inria `.ply`.
pub fn encode_ply(scene: &Scene) -> Result<Vec<u8>, PlyError> {
    encode_ply_with(scene, &[])
}

/// [`encode_ply`] with extra float properties after the Inria ones, `(name,
/// one value per gaussian)`, such as a class per gaussian. Readers that do not
/// know them (this one included) skip them.
pub fn encode_ply_with(scene: &Scene, extra: &[(&str, &[f32])]) -> Result<Vec<u8>, PlyError> {
    if scene.is_empty() {
        return Err(PlyError::NothingToWrite);
    }
    if let Some((name, values)) = extra.iter().find(|(_, v)| v.len() != scene.len()) {
        return Err(PlyError::ExtraLength {
            name: (*name).to_owned(),
            values: values.len(),
            gaussians: scene.len(),
        });
    }
    let degree = scene.sh_degree;
    let coeffs = sh_rest_coeffs_per_channel(degree);
    let rest_total = coeffs * 3;
    // x,y,z,nx,ny,nz,dc0..2,rest...,op,scale0..2,rot0..3, extra...
    let stride = 17 + rest_total + extra.len();

    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {}\n", scene.len()));
    for name in ["x", "y", "z", "nx", "ny", "nz"] {
        header.push_str(&format!("property float {name}\n"));
    }
    for c in 0..3 {
        header.push_str(&format!("property float f_dc_{c}\n"));
    }
    for k in 0..rest_total {
        header.push_str(&format!("property float f_rest_{k}\n"));
    }
    header.push_str("property float opacity\n");
    for c in 0..3 {
        header.push_str(&format!("property float scale_{c}\n"));
    }
    for c in 0..4 {
        header.push_str(&format!("property float rot_{c}\n"));
    }
    for (name, _) in extra {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");

    let mut out = Vec::with_capacity(header.len() + scene.len() * stride * 4);
    out.extend_from_slice(header.as_bytes());
    let mut push = |v: f32| out.extend_from_slice(&v.to_le_bytes());
    for (i, g) in scene.gaussians.iter().enumerate() {
        push(g.mean.x);
        push(g.mean.y);
        push(g.mean.z);
        push(0.0);
        push(0.0);
        push(0.0);
        push(g.sh_dc[0]);
        push(g.sh_dc[1]);
        push(g.sh_dc[2]);
        for k in 0..rest_total {
            push(g.sh_rest.get(k).copied().unwrap_or(0.0));
        }
        push(g.opacity_logit);
        push(g.scale_log.x);
        push(g.scale_log.y);
        push(g.scale_log.z);
        push(g.rotation.w);
        push(g.rotation.i);
        push(g.rotation.j);
        push(g.rotation.k);
        for (_, values) in extra {
            push(values[i]);
        }
    }
    Ok(out)
}

/// Write a [`Scene`] as a binary-little-endian Inria `.ply`.
pub fn save_ply(scene: &Scene, path: impl AsRef<Path>) -> Result<(), PlyError> {
    fs::write(path, encode_ply(scene)?)?;
    Ok(())
}

/// [`save_ply`] with extra float properties (see [`encode_ply_with`]).
pub fn save_ply_with(
    scene: &Scene,
    extra: &[(&str, &[f32])],
    path: impl AsRef<Path>,
) -> Result<(), PlyError> {
    fs::write(path, encode_ply_with(scene, extra)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(degree: u32) -> Scene {
        let coeffs = sh_rest_coeffs_per_channel(degree);
        let sh_rest: Vec<f32> = (0..3 * coeffs).map(|i| i as f32 * 0.01).collect();
        Scene::new(
            vec![Gaussian {
                mean: Vector3::new(1.5, -0.25, 4.0),
                scale_log: Vector3::new(-1.0, 0.5, 2.0),
                rotation: Quaternion::new(0.5, 0.5, 0.5, 0.5),
                opacity_logit: 1.25,
                sh_dc: [0.1, -0.2, 0.3],
                sh_rest,
                sh_degree: degree,
            }],
            degree,
        )
    }

    #[test]
    fn round_trips_degree_3() {
        let scene = sample(3);
        let bytes = encode_ply(&scene).unwrap();
        let decoded = scene_from_ply(&bytes).unwrap();
        assert_eq!(decoded.sh_degree, 3);
        assert_eq!(decoded.len(), 1);
        let a = &scene.gaussians[0];
        let b = &decoded.gaussians[0];
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.scale_log, b.scale_log);
        assert_eq!(a.opacity_logit, b.opacity_logit);
        assert_eq!(a.sh_dc, b.sh_dc);
        assert_eq!(a.sh_rest.len(), b.sh_rest.len());
        for (x, y) in a.sh_rest.iter().zip(&b.sh_rest) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn round_trips_degree_0() {
        let scene = sample(0);
        let bytes = encode_ply(&scene).unwrap();
        let decoded = scene_from_ply(&bytes).unwrap();
        assert_eq!(decoded.sh_degree, 0);
        assert!(decoded.gaussians[0].sh_rest.is_empty());
        assert_eq!(decoded.gaussians[0].mean, scene.gaussians[0].mean);
    }

    #[test]
    fn extra_properties_follow_and_are_skipped_on_read() {
        let scene = sample(1);
        let bytes = encode_ply_with(&scene, &[("class", &[3.0]), ("confidence", &[0.5])]).unwrap();
        let header = String::from_utf8_lossy(&bytes[..bytes.len() - 4 * (17 + 9 + 2)]);
        assert!(header.ends_with("property float class\nproperty float confidence\nend_header\n"));
        assert_eq!(
            &bytes[bytes.len() - 8..],
            [3.0f32.to_le_bytes(), 0.5f32.to_le_bytes()].concat()
        );
        assert_eq!(scene_from_ply(&bytes).unwrap(), scene);
        assert!(matches!(
            encode_ply_with(&scene, &[("class", &[])]),
            Err(PlyError::ExtraLength { .. })
        ));
    }

    #[test]
    fn extra_properties_read_back_by_name() {
        let first = sample(2).gaussians[0].clone();
        let mut second = first.clone();
        second.opacity_logit = -0.75;
        let scene = Scene::new(vec![first, second], 2);
        let bytes = encode_ply_with(&scene, &[("a", &[1.0, 2.0]), ("b", &[-3.0, 4.5])]).unwrap();
        let got = ply_properties(&bytes, &["b", "missing", "a", "opacity"]).unwrap();
        assert_eq!(got[0], Some(vec![-3.0, 4.5]));
        assert_eq!(got[1], None);
        assert_eq!(got[2], Some(vec![1.0, 2.0]));
        let opacity: Vec<f32> = scene.gaussians.iter().map(|g| g.opacity_logit).collect();
        assert_eq!(got[3], Some(opacity));
    }

    #[test]
    fn rejects_non_ply() {
        assert!(matches!(scene_from_ply(b"nope"), Err(PlyError::BadMagic)));
    }

    #[test]
    fn rejects_empty_scene() {
        assert!(matches!(
            encode_ply(&Scene::new(vec![], 0)),
            Err(PlyError::NothingToWrite)
        ));
    }
}
