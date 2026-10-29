//! IMU and video frame times from an Insta360 `.insv` file's trailer.
//!
//! The camera appends a trailer to the MP4: it ends with the ASCII magic
//! `8db42d694ccc418790edff439fe026bf`, preceded by the version (u32) and the
//! trailer's size (u32), all little-endian. Just before that 72-byte footer
//! sits an index record (6-byte header `format u8, id u8, size u32` after its
//! data) whose entries `id u16, size u32, offset u32` (offset from the
//! trailer's start) locate the other records. Two are read here:
//! - id 3, the IMU: per sample a u64 timestamp in microseconds and six i16
//!   counts in offset binary (sign bit flipped), accelerometer then gyro in
//!   one sensor frame, at the X5's 1024 counts per g and 16.4 counts per
//!   deg/s (docs/insta360_imu_scale.md; checked on an X5, firmware v1.11.6).
//! - id 4, exposure: per video frame a u64 timestamp in microseconds (the
//!   IMU's clock) and the exposure time as f64 seconds.
//!
//! Times are returned in seconds from the first video frame. Only the file's
//! end is read, so the trailer alone (its last bytes) also works.
//!
//! The metadata record (id 0x101) also holds each lens's factory
//! calibration as `_`-separated text ([`InsvLens`]; docs/insta360_lens.md).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use nalgebra::Vector3;

use crate::imu_csv::{
    ImuCsv, INSTA360_ACCEL_COUNTS_PER_G, INSTA360_GYRO_COUNTS_PER_DPS, STANDARD_GRAVITY,
};

pub const INSV_MAGIC: &[u8; 32] = b"8db42d694ccc418790edff439fe026bf";
const FOOTER_LEN: usize = 72;
const RECORD_IMU: u16 = 3;
const RECORD_EXPOSURE: u16 = 4;
const RECORD_METADATA: u16 = 0x101;

/// What [`read_insv_telemetry`] returns.
#[derive(Debug, Clone, PartialEq)]
pub struct InsvTelemetry {
    /// The camera model in the metadata record (e.g. "Insta360 X5"), if found.
    pub camera: Option<String>,
    /// IMU samples, seconds from the first video frame, rad/s and m/s².
    pub imu: ImuCsv,
    /// Each video frame's timestamp, seconds from the first.
    pub frame_times: Vec<f64>,
    /// Each video frame's exposure time, seconds.
    pub exposure: Vec<f64>,
    /// The lenses' calibration, front lens (the first video stream) first;
    /// empty when the metadata has none in the layout [`InsvLens`] reads.
    pub lenses: Vec<InsvLens>,
}

/// One lens's factory calibration from the metadata record, in the pixels of
/// the calibration's full-resolution lens image (5376 square on an X5).
///
/// The unified (Mei) model: a ray `(x, y, z)` in the lens frame (x right,
/// y down, z along the optical axis) goes to `(x, y) / (z + xi |ray|)`,
/// then radial distortion `1 + k1 r² + k2 r⁴ + k3 r⁶` and tangential
/// `p1, p2` (OpenCV's), then `fx, fy, cx, cy`. On an X5 the video holds
/// each lens's image upside down relative to the calibration, centred on
/// the middle of the lens's calibration square rather than on `(cx, cy)`
/// (`visloc_gsplat_train::dual_fisheye::FisheyeLens::from_insv`; measured
/// from where the two lenses see the same things, docs/insta360_lens.md).
#[derive(Debug, Clone, PartialEq)]
pub struct InsvLens {
    pub xi: f64,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub radial: [f64; 3],
    pub tangential: [f64; 2],
    /// The radius of the 200° image circle.
    pub circle_radius: f64,
    /// The lens's orientation on the camera as recorded (three angles in
    /// degrees; their convention is not decoded).
    pub angles: [f64; 3],
    /// The lens's position on the camera as recorded (metres; the back lens
    /// sits about 3 cm behind the front one).
    pub translation: [f64; 3],
    /// The calibration image's width and height: both lenses side by side
    /// (10752 x 5376 on an X5), so the lens's own square is the half that
    /// holds `(cx, cy)`.
    pub calibration_size: [f64; 2],
}

#[derive(Debug, thiserror::Error)]
pub enum InsvError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: no Insta360 trailer (magic missing)")]
    NoTrailer { path: PathBuf },
    #[error("{path}: invalid Insta360 trailer: {message}")]
    Invalid { path: PathBuf, message: String },
}

/// Reads the IMU and the video frame times from an `.insv` file's trailer.
pub fn read_insv_telemetry(path: &Path) -> Result<InsvTelemetry, InsvError> {
    let io = |source| InsvError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let len = file.metadata().map_err(io)?.len();
    if len < FOOTER_LEN as u64 {
        return Err(InsvError::NoTrailer {
            path: path.to_path_buf(),
        });
    }
    let mut footer = [0u8; FOOTER_LEN];
    file.seek(SeekFrom::End(-(FOOTER_LEN as i64))).map_err(io)?;
    file.read_exact(&mut footer).map_err(io)?;
    if &footer[40..] != INSV_MAGIC {
        return Err(InsvError::NoTrailer {
            path: path.to_path_buf(),
        });
    }
    let size = u32::from_le_bytes(footer[32..36].try_into().unwrap()) as u64;
    if size > len || (size as usize) < FOOTER_LEN + 6 {
        return Err(invalid(
            path,
            format!("trailer size {size} for a {len}-byte file"),
        ));
    }
    let mut trailer = vec![0u8; size as usize];
    file.seek(SeekFrom::End(-(size as i64))).map_err(io)?;
    file.read_exact(&mut trailer).map_err(io)?;
    parse_trailer(&trailer, path)
}

fn invalid(path: &Path, message: String) -> InsvError {
    InsvError::Invalid {
        path: path.to_path_buf(),
        message,
    }
}

/// Parses a trailer (the file's last `size` bytes, footer included).
pub fn parse_trailer(trailer: &[u8], path: &Path) -> Result<InsvTelemetry, InsvError> {
    let records = trailer_records(trailer).map_err(|m| invalid(path, m))?;
    let find = |id: u16| {
        records
            .iter()
            .find(|(r, _)| *r == id)
            .map(|(_, data)| *data)
    };
    let exposure =
        find(RECORD_EXPOSURE).ok_or_else(|| invalid(path, "no exposure record".into()))?;
    if exposure.len() % 16 != 0 || exposure.is_empty() {
        return Err(invalid(
            path,
            format!("exposure record of {} bytes", exposure.len()),
        ));
    }
    let frames: Vec<(u64, f64)> = exposure
        .chunks_exact(16)
        .map(|c| {
            (
                u64::from_le_bytes(c[..8].try_into().unwrap()),
                f64::from_le_bytes(c[8..].try_into().unwrap()),
            )
        })
        .collect();
    let origin = frames[0].0;
    let seconds = |us: u64| (us as f64 - origin as f64) * 1e-6;

    let raw = find(RECORD_IMU).ok_or_else(|| invalid(path, "no IMU record".into()))?;
    if raw.len() % 20 != 0 || raw.is_empty() {
        return Err(invalid(
            path,
            format!(
                "IMU record of {} bytes is not 20-byte int16 samples",
                raw.len()
            ),
        ));
    }
    let count = |c: &[u8]| (u16::from_le_bytes([c[0], c[1]]) ^ 0x8000) as i16 as f64;
    let gyro_scale = (1.0 / INSTA360_GYRO_COUNTS_PER_DPS).to_radians();
    let accel_scale = STANDARD_GRAVITY / INSTA360_ACCEL_COUNTS_PER_G;
    let mut imu = ImuCsv::default();
    for s in raw.chunks_exact(20) {
        imu.time
            .push(seconds(u64::from_le_bytes(s[..8].try_into().unwrap())));
        let v: Vec<f64> = s[8..].chunks_exact(2).map(count).collect();
        imu.accel.push(Vector3::new(v[0], v[1], v[2]) * accel_scale);
        imu.gyro.push(Vector3::new(v[3], v[4], v[5]) * gyro_scale);
    }
    let metadata = find(RECORD_METADATA).map(String::from_utf8_lossy);
    let camera = metadata.as_deref().and_then(|text| {
        let start = text.find("Insta360")?;
        let name: String = text[start..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == ' ')
            .collect();
        Some(name.trim().to_string())
    });
    Ok(InsvTelemetry {
        camera,
        imu,
        frame_times: frames.iter().map(|&(t, _)| seconds(t)).collect(),
        exposure: frames.iter().map(|&(_, e)| e).collect(),
        lenses: metadata.as_deref().map(parse_lenses).unwrap_or_default(),
    })
}

/// The lenses in the metadata's text: the unified-model string
/// `2_` then per lens `xi_fx_fy_cx_cy_` three angles, three translations,
/// `k1_k2_k3_p1_p2_` and the calibration image's width, height and a code
/// (19 fields; then one more field), and the image circles' radii from the
/// string after a letter and `2_` (`p2_` or `n2_` on X5s, per lens
/// `radius_cx_cy_` three angles), matched by centre.
fn parse_lenses(text: &str) -> Vec<InsvLens> {
    const PER_LENS: usize = 19;
    let runs = || {
        text.split(|c: char| !c.is_ascii_graphic())
            .filter(|r| r.len() > 20)
    };
    let numbers = |s: &str| -> Option<Vec<f64>> { s.split('_').map(|t| t.parse().ok()).collect() };
    // Every `<letter>2_` string's per-lens `radius cx cy`, in order; a lens
    // takes the first whose centre is its own.
    let circles: Vec<[f64; 3]> = runs()
        .flat_map(|r| {
            r.match_indices("2_").filter_map(move |(i, _)| {
                if !r[..i].ends_with(|c: char| c.is_ascii_alphabetic()) {
                    return None;
                }
                let v = numbers(&r[i + 2..])?;
                (v.len() >= 12).then(|| [[v[0], v[1], v[2]], [v[6], v[7], v[8]]])
            })
        })
        .flatten()
        .collect();
    // The string may follow a printable length byte of the record's framing.
    let Some(v) = runs().find_map(|r| {
        r.match_indices("2_").find_map(|(i, _)| {
            if r[..i].ends_with(|c: char| c.is_ascii_digit() || c == '.' || c == '-') {
                return None;
            }
            let v = numbers(&r[i..])?;
            (v.len() == 2 + 2 * PER_LENS).then_some(v)
        })
    }) else {
        return Vec::new();
    };
    let lenses: Option<Vec<InsvLens>> = (0..2)
        .map(|i| {
            let b = &v[1 + i * PER_LENS..1 + (i + 1) * PER_LENS];
            let (cx, cy) = (b[3], b[4]);
            let circle = circles
                .iter()
                .find(|c| (c[1] - cx).abs() < 1.0 && (c[2] - cy).abs() < 1.0)?;
            Some(InsvLens {
                xi: b[0],
                fx: b[1],
                fy: b[2],
                cx,
                cy,
                angles: [b[5], b[6], b[7]],
                translation: [b[8], b[9], b[10]],
                radial: [b[11], b[12], b[13]],
                tangential: [b[14], b[15]],
                circle_radius: circle[0],
                calibration_size: [b[16], b[17]],
            })
        })
        .collect();
    lenses.unwrap_or_default()
}

/// `(id, data)` of every record in the trailer's index.
fn trailer_records(trailer: &[u8]) -> Result<Vec<(u16, &[u8])>, String> {
    let n = trailer.len();
    if n < FOOTER_LEN + 6 || &trailer[n - 32..] != INSV_MAGIC {
        return Err("magic missing".into());
    }
    let header = &trailer[n - FOOTER_LEN - 6..n - FOOTER_LEN];
    let index_len = u32::from_le_bytes(header[2..6].try_into().unwrap()) as usize;
    let end = n - FOOTER_LEN - 6;
    let index = trailer
        .get(
            end.checked_sub(index_len)
                .ok_or("index record past the trailer's start")?..end,
        )
        .ok_or("index record out of range")?;
    let mut out = Vec::new();
    for e in index.chunks_exact(10) {
        let id = u16::from_le_bytes([e[0], e[1]]);
        let size = u32::from_le_bytes(e[2..6].try_into().unwrap()) as usize;
        let offset = u32::from_le_bytes(e[6..10].try_into().unwrap()) as usize;
        if size == 0 {
            continue;
        }
        let data = trailer
            .get(offset..offset + size)
            .ok_or_else(|| format!("record {id:#x} ({size} bytes at {offset}) out of range"))?;
        out.push((id, data));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trailer with the given records, laid out as the camera does: the
    /// records, the index record and its header, then the footer.
    fn trailer(records: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut body = Vec::new();
        let mut index = Vec::new();
        for (id, data) in records {
            index.extend_from_slice(&id.to_le_bytes());
            index.extend_from_slice(&(data.len() as u32).to_le_bytes());
            index.extend_from_slice(&(body.len() as u32).to_le_bytes());
            body.extend_from_slice(data);
        }
        index.extend_from_slice(&[0u8; 10]);
        body.extend_from_slice(&index);
        body.extend_from_slice(&[0, 0]);
        body.extend_from_slice(&(index.len() as u32).to_le_bytes());
        body.extend_from_slice(&[0u8; 32]);
        let size = (body.len() + 40) as u32;
        body.extend_from_slice(&size.to_le_bytes());
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(INSV_MAGIC);
        body
    }

    fn offset_binary(v: i16) -> [u8; 2] {
        ((v as u16) ^ 0x8000).to_le_bytes()
    }

    #[test]
    fn reads_imu_and_frame_times() {
        let mut imu = Vec::new();
        for k in 0..3u64 {
            imu.extend_from_slice(&(1_000_000 + 1000 * k).to_le_bytes());
            for v in [0i16, -1024, 0, 164, -328, 0] {
                imu.extend_from_slice(&offset_binary(v));
            }
        }
        let mut exposure = Vec::new();
        for k in 0..4u64 {
            exposure.extend_from_slice(&(1_000_500 + 33_368 * k).to_le_bytes());
            exposure.extend_from_slice(&0.0005f64.to_le_bytes());
        }
        let mut meta = b"\x0a\x0eSERIAL\x12\x0bInsta360 X5\x1a".to_vec();
        meta.extend_from_slice(&[0u8; 4]);
        let data = trailer(&[
            (RECORD_METADATA, meta),
            (RECORD_IMU, imu),
            (RECORD_EXPOSURE, exposure),
        ]);
        let dir = std::env::temp_dir().join(format!("insv_{}.insv", std::process::id()));
        let mut file = vec![0u8; 100]; // stands in for the MP4
        file.extend_from_slice(&data);
        std::fs::write(&dir, &file).unwrap();
        let t = read_insv_telemetry(&dir).unwrap();
        std::fs::remove_file(&dir).unwrap();
        assert_eq!(t.camera.as_deref(), Some("Insta360 X5"));
        assert_eq!(t.frame_times.len(), 4);
        assert!((t.frame_times[1] - 0.033368).abs() < 1e-12);
        assert_eq!(t.exposure[0], 0.0005);
        assert_eq!(t.imu.time.len(), 3);
        assert!((t.imu.time[0] + 0.0005).abs() < 1e-12);
        assert!((t.imu.accel[0].y + STANDARD_GRAVITY).abs() < 1e-12);
        assert!((t.imu.gyro[0].x - 10f64.to_radians()).abs() < 1e-12);
        assert!((t.imu.gyro[0].y + 20f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn parses_the_lens_calibration() {
        // The X5's layout with made-up values; a printable byte ('J', a
        // length) right before the unified-model string.
        let text = "\u{a}\u{e}SERIAL\u{12}\u{b}Insta360 X5\u{1a}\u{5}v1.0*\
            p2_2650.000_2700.000_2680.000_0.400_-0.200_89.800_\
            2651.000_8080.000_2675.000_-0.400_-0.200_91.100_10752_5376_11111\u{0}\
            \u{2a}J2_2.000000_4270.000_4271.000_2700.000_2680.000_0.400_-0.200_89.800_\
            0.000000_0.000000_0.000000_0.18_2.08_-3.2_-0.0003_-0.0016_10752_5376_113_\
            2.000000_4274.000_4275.000_8080.000_2675.000_-0.400_-0.200_91.100_\
            -0.001_0.0004_-0.032_0.17_2.1_-3.3_-0.0015_0.0004_10752_5376_113_197632\u{0}";
        let lenses = parse_lenses(text);
        assert_eq!(lenses.len(), 2);
        let (front, back) = (&lenses[0], &lenses[1]);
        assert_eq!((front.xi, front.fx, front.fy), (2.0, 4270.0, 4271.0));
        assert_eq!(
            (front.cx, front.cy, front.circle_radius),
            (2700.0, 2680.0, 2650.0)
        );
        assert_eq!(front.radial, [0.18, 2.08, -3.2]);
        assert_eq!(front.tangential, [-0.0003, -0.0016]);
        assert_eq!(back.cx, 8080.0);
        assert_eq!(back.circle_radius, 2651.0);
        assert_eq!(back.translation, [-0.001, 0.0004, -0.032]);
        assert_eq!(back.angles, [-0.4, -0.2, 91.1]);
        assert_eq!(back.calibration_size, [10752.0, 5376.0]);
        // Some files write `n2_` instead of `p2_`.
        assert_eq!(parse_lenses(&text.replace("p2_", "n2_")), lenses);
        // Without the circles' string there is no lens image to map to.
        let no_circles = text.replace("p2_", "*2_");
        assert!(parse_lenses(&no_circles).is_empty());
    }

    #[test]
    fn rejects_a_file_without_trailer() {
        let dir = std::env::temp_dir().join(format!("insv_none_{}.insv", std::process::id()));
        std::fs::write(&dir, vec![1u8; 500]).unwrap();
        let err = read_insv_telemetry(&dir).unwrap_err();
        std::fs::remove_file(&dir).unwrap();
        assert!(matches!(err, InsvError::NoTrailer { .. }));
    }
}
