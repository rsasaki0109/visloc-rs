//! Synthetic sensor data shared by the integration tests.
#![allow(dead_code)]

use std::path::PathBuf;

use nalgebra::{Point3, Vector3};
use visloc_basalt::ImuSample;
use visloc_core::types::{Camera, Landmark, LandmarkDescriptorStore, VisualMap};
use visloc_ros2::image::{decode_luma, mono8_image};
use visloc_ros2::localize::{LocalizeCore, LocalizeOptions};
use visloc_ros2::msgs::{Header, Image, Imu, Time, Vector3 as RosVector3};
use visloc_vision::features::sift::{extract_sift_features, GrayImage};

pub const CAMERA_PERIOD_NS: i64 = 50_000_000; // 20 Hz
pub const IMU_PERIOD_NS: i64 = 5_000_000; // 200 Hz
pub const START_NS: i64 = 1_700_000_000_000_000_000;

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The Basalt EuRoC calibration/config pair the online demo documents.
pub fn euroc_calibration_path() -> PathBuf {
    repo_root().join("configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json")
}

pub fn euroc_config_path() -> PathBuf {
    repo_root().join("configs/basalt/variants/official_euroc_ds/euroc_config.json")
}

/// Deterministic multi-octave value noise in `[0, 255]`: well textured at
/// several scales, so both FAST/KLT and SIFT find plenty of structure.
pub fn texture(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let hash = |x: i64, y: i64, octave: u64| -> f64 {
        let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9)
            ^ octave.wrapping_mul(0x27D4_EB2F_1656_67C5);
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        (h >> 11) as f64 / (1u64 << 53) as f64
    };
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let mut raw = vec![0.0f64; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            let mut weight_sum = 0.0;
            for (octave, cell) in [(0u64, 48.0), (1, 23.0), (2, 11.0), (3, 5.0)] {
                let fx = x as f64 / cell;
                let fy = y as f64 / cell;
                let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
                let (tx, ty) = (smooth(fx - x0 as f64), smooth(fy - y0 as f64));
                let a = hash(x0, y0, octave);
                let b = hash(x0 + 1, y0, octave);
                let c = hash(x0, y0 + 1, octave);
                let d = hash(x0 + 1, y0 + 1, octave);
                let v = (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty;
                let weight = 1.0 / (octave as f64 + 1.0);
                value += v * weight;
                weight_sum += weight;
            }
            raw[y * width + x] = value / weight_sum;
        }
    }
    // Averaging octaves shrinks the dynamic range; stretch to full 8 bits.
    let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    raw.iter()
        .map(|v| ((v - min) / (max - min) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// `out(x, y) = src(x + shift, y)` with edge clamping.
pub fn shift_horizontally(src: &[u8], width: usize, height: usize, shift: i64) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let sx = (x as i64 + shift).clamp(0, width as i64 - 1) as usize;
            out[y * width + x] = src[y * width + sx];
        }
    }
    out
}

/// A static textured stereo rig: right = left shifted by a constant
/// disparity (a fronto-parallel plane in front of the rig).
pub struct SyntheticStereo {
    pub width: usize,
    pub height: usize,
    pub left: Vec<u8>,
    pub right: Vec<u8>,
}

impl SyntheticStereo {
    pub fn euroc_sized() -> Self {
        let (width, height) = (752, 480);
        let left = texture(width, height, 7);
        let right = shift_horizontally(&left, width, height, 12);
        Self {
            width,
            height,
            left,
            right,
        }
    }

    pub fn image_msg(&self, right: bool, stamp_ns: i64) -> Image {
        let data = if right { &self.right } else { &self.left };
        mono8_image(
            Header::new(
                Time::from_nanos(stamp_ns),
                if right { "cam1" } else { "cam0" },
            ),
            self.width as u32,
            self.height as u32,
            data.clone(),
        )
    }
}

/// A stationary IMU: zero rate, specific force pointing "up" along +x as
/// in EuRoC's IMU mounting.
pub fn stationary_imu_msg(stamp_ns: i64) -> Imu {
    Imu {
        header: Header::new(Time::from_nanos(stamp_ns), "imu0"),
        linear_acceleration: RosVector3 {
            x: 9.81,
            y: 0.0,
            z: 0.0,
        },
        ..Imu::default()
    }
}

pub fn stationary_imu_sample(stamp_ns: i64) -> ImuSample {
    ImuSample::new(stamp_ns, Vector3::zeros(), Vector3::new(9.81, 0.0, 0.0))
}

/// Pinhole camera of the synthetic localization scene.
pub const LOC_WIDTH: usize = 320;
pub const LOC_HEIGHT: usize = 240;
pub const LOC_FOCAL: f64 = 300.0;
/// The scene is three fronto-parallel slabs, one per horizontal image band
/// (rows `[0, 80)`, `[80, 160)`, `[160, 240)`), at these depths. A planar
/// scene would be degenerate for the default DLT-PnP minimal solver.
pub const BAND_DEPTHS: [f64; 3] = [4.0, 6.0, 8.0];
pub const BAND_ROWS: usize = 80;
/// Query camera translation along +x (m): shifts the bands by exactly
/// `f * tx / Z` = 12, 8 and 6 pixels.
pub const QUERY_TX: f64 = 0.16;

fn band_depth(row: f64) -> f64 {
    BAND_DEPTHS[((row.max(0.0) as usize) / BAND_ROWS).min(BAND_DEPTHS.len() - 1)]
}

pub fn loc_camera() -> Camera {
    Camera::pinhole(
        1,
        LOC_WIDTH as u32,
        LOC_HEIGHT as u32,
        LOC_FOCAL,
        LOC_FOCAL,
        LOC_WIDTH as f64 / 2.0,
        LOC_HEIGHT as f64 / 2.0,
    )
}

/// Value noise plus a few hundred random bright/dark Gaussian blobs of
/// several sizes: blob-like structure is what DoG/SIFT detects reliably.
pub fn loc_map_image() -> Vec<u8> {
    let (width, height) = (LOC_WIDTH, LOC_HEIGHT);
    let base = texture(width, height, 21);
    let mut image: Vec<f64> = base.iter().map(|&v| 0.5 * f64::from(v)).collect();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..400 {
        let cx = next() * width as f64;
        let cy = next() * height as f64;
        let sigma = 1.5 + next() * 4.0;
        let amplitude = if next() < 0.5 { -110.0 } else { 110.0 };
        let reach = (3.0 * sigma).ceil() as i64;
        for y in (cy as i64 - reach).max(0)..(cy as i64 + reach).min(height as i64) {
            for x in (cx as i64 - reach).max(0)..(cx as i64 + reach).min(width as i64) {
                let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                image[y as usize * width + x as usize] +=
                    amplitude * (-d2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    image
        .iter()
        .map(|v| (v + 64.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// The query rendered from a camera translated by `QUERY_TX` along x: for
/// each fronto-parallel band that is exactly a horizontal image shift.
pub fn loc_query_image() -> Vec<u8> {
    let src = loc_map_image();
    let mut out = vec![0u8; LOC_WIDTH * LOC_HEIGHT];
    for y in 0..LOC_HEIGHT {
        let shift = (LOC_FOCAL * QUERY_TX / band_depth(y as f64)).round() as i64;
        let row = shift_horizontally(
            &src[y * LOC_WIDTH..(y + 1) * LOC_WIDTH],
            LOC_WIDTH,
            1,
            shift,
        );
        out[y * LOC_WIDTH..(y + 1) * LOC_WIDTH].copy_from_slice(&row);
    }
    out
}

pub fn expected_query_center_x() -> f64 {
    QUERY_TX
}

/// Builds a tiny map: SIFT on the map image, every keypoint back-projected
/// onto its band's slab depth, seen by a camera at the origin.
pub fn synthetic_map(options: &LocalizeOptions) -> (VisualMap, LandmarkDescriptorStore) {
    let msg = mono8_image(
        Header::default(),
        LOC_WIDTH as u32,
        LOC_HEIGHT as u32,
        loc_map_image(),
    );
    let pixels = decode_luma(&msg).unwrap().to_unit_f32();
    let gray = GrayImage::new(LOC_WIDTH, LOC_HEIGHT, &pixels).unwrap();
    let features = extract_sift_features(&gray, &options.sift).unwrap();
    assert!(
        features.keypoints.len() > 50,
        "texture should give SIFT features, got {}",
        features.keypoints.len()
    );
    let camera = loc_camera();
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut store = LandmarkDescriptorStore::new();
    for (index, (keypoint, descriptor)) in features
        .keypoints
        .iter()
        .zip(features.descriptors.iter())
        .enumerate()
    {
        let normalized = camera.normalize_pixel(keypoint).unwrap();
        let depth = band_depth(keypoint.y);
        let id = index as u64 + 1;
        map.landmarks.insert(
            id,
            Landmark::new(
                id,
                Point3::new(normalized.x * depth, normalized.y * depth, depth),
            ),
        );
        store.insert(id, descriptor.clone());
    }
    (map, store)
}

/// Writes a descriptor store in `read_landmark_descriptors_txt` format.
pub fn write_descriptor_store(store: &LandmarkDescriptorStore, path: &std::path::Path) -> PathBuf {
    use std::fmt::Write as _;
    let mut text = String::new();
    let mut entries: Vec<_> = store.iter().collect();
    entries.sort_by_key(|(id, _)| *id);
    for (id, descriptor) in entries {
        write!(text, "{id}").unwrap();
        for value in descriptor {
            write!(text, " {value}").unwrap();
        }
        text.push('\n');
    }
    std::fs::write(path, text).unwrap();
    path.to_path_buf()
}

/// Writes the synthetic map as a COLMAP text model + descriptor store into
/// `dir`; returns the descriptor store path.
pub fn write_synthetic_map(dir: &std::path::Path) -> PathBuf {
    let (map, store) = synthetic_map(&LocalizeOptions::default());
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    visloc_io::colmap::write_colmap_text_model(&map, dir).unwrap();
    write_descriptor_store(&store, &dir.join("landmark_descriptors.txt"))
}

pub fn synthetic_localize_core() -> LocalizeCore {
    let options = LocalizeOptions::default();
    let (map, store) = synthetic_map(&options);
    LocalizeCore::new(map, store, Some(1), options).unwrap()
}
