//! Sensor-only EuRoC MAV reader for the dedicated Basalt pipeline.
//!
//! This reader deliberately knows only about `mav0/cam0`, `mav0/cam1`, and
//! `mav0/imu0`.  It never opens a ground-truth, mocap, Leica, or state
//! directory.  EuRoC's usual 8-bit grayscale PNGs are placed in Basalt's raw
//! `u16` container with the upstream conversion `u16 = u8 << 8`; genuine
//! 16-bit grayscale PNGs are copied without rescaling.

use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};

use image::DynamicImage;
use nalgebra::Vector3;
use thiserror::Error;

use crate::{
    config::{BasaltConfig, ConfigError},
    pyramid::{ImageError as RawImageError, RawU16Image},
    timing::{TimingBreakdown, TimingBucket},
    BasaltCalibration, CalibrationError, FrameId, ImuSample, TimestampNs,
};

/// One row from a EuRoC camera `data.csv` manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EurocImageEntry {
    pub timestamp_ns: TimestampNs,
    pub filename: String,
    pub path: PathBuf,
}

/// One decoded stereo/IMU event delivered to the Basalt adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocSensorFrame {
    pub frame_id: FrameId,
    pub timestamp_ns: TimestampNs,
    pub cam0: RawU16Image,
    pub cam1: Option<RawU16Image>,
    pub cam0_path: PathBuf,
    pub cam1_path: Option<PathBuf>,
    /// Samples strictly after the preceding camera timestamp and through this
    /// frame timestamp.  For frame zero, all samples at or before its camera
    /// timestamp are returned.
    pub imu: Vec<ImuSample>,
    /// First IMU sample at or after the first camera timestamp.  Upstream
    /// Basalt skips older packets and uses this sample only to align the
    /// initial pose with gravity; normal interval delivery remains in `imu`.
    pub initialization_imu: Option<ImuSample>,
}

/// EuRoC sensor data and the two upstream Basalt JSON contracts.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocSensorDataset {
    root: PathBuf,
    cam0_images: Vec<EurocImageEntry>,
    cam0_manifest_count: usize,
    cam1_by_timestamp: BTreeMap<TimestampNs, EurocImageEntry>,
    imu_samples: Vec<ImuSample>,
    calibration: BasaltCalibration,
    config: BasaltConfig,
    monocular: bool,
}

impl EurocSensorDataset {
    /// Opens a sensor-only EuRoC recording and the pinned Basalt JSON inputs.
    ///
    /// Only manifests and IMU values are read here; PNG pixels are decoded on
    /// demand by [`Self::frame`], keeping an 80-frame smoke run bounded.
    pub fn open(
        root: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self, EurocReaderError> {
        // Keep the long-standing API's normal path completely independent of
        // process timing configuration.  The demo uses `open_with_timing`
        // so one collector can include setup and per-frame acquisition.
        let mut timing = TimingBreakdown::default();
        Self::open_with_timing(root, calibration_path, config_path, &mut timing)
    }

    /// Opens the sensor-only recording while optionally collecting the
    /// diagnostic setup buckets.  The collector is opt-in and does not alter
    /// parsed values, ordering, or error propagation.
    pub fn open_with_timing(
        root: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        timing: &mut TimingBreakdown,
    ) -> Result<Self, EurocReaderError> {
        let started = timing.start();
        let result = Self::open_impl(root, calibration_path, config_path, false, timing);
        timing.finish(TimingBucket::DatasetOpen, started);
        result
    }

    /// Opens the recording as a monocular-inertial (cam0 + IMU) stream.
    ///
    /// `mav0/cam1` is never read (it may be absent), every cam0 manifest row
    /// becomes a frame, every [`EurocSensorFrame::cam1`] is `None`, and the
    /// calibration is reduced to camera 0 with
    /// [`BasaltCalibration::retain_cameras`]. A stereo calibration file can
    /// therefore be reused unchanged. The stereo [`Self::open`] path is not
    /// affected by this mode.
    pub fn open_monocular(
        root: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self, EurocReaderError> {
        let mut timing = TimingBreakdown::default();
        Self::open_monocular_with_timing(root, calibration_path, config_path, &mut timing)
    }

    /// [`Self::open_monocular`] with the optional setup timing buckets.
    pub fn open_monocular_with_timing(
        root: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        timing: &mut TimingBreakdown,
    ) -> Result<Self, EurocReaderError> {
        let started = timing.start();
        let result = Self::open_impl(root, calibration_path, config_path, true, timing);
        timing.finish(TimingBucket::DatasetOpen, started);
        result
    }

    fn open_impl(
        root: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        monocular: bool,
        timing: &mut TimingBreakdown,
    ) -> Result<Self, EurocReaderError> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(EurocReaderError::MissingDirectory(root));
        }
        let mav0 = root.join("mav0");
        let cam0_dir = mav0.join("cam0");
        let cam1_dir = mav0.join("cam1");
        let imu0_dir = mav0.join("imu0");

        let mut cam0_images = timing.measure(TimingBucket::DatasetCsvParsing, || {
            read_image_manifest(&cam0_dir.join("data.csv"), &cam0_dir)
        })?;
        if cam0_images.is_empty() {
            return Err(EurocReaderError::EmptyManifest(cam0_dir.join("data.csv")));
        }
        // Monocular mode never touches cam1: no manifest, no intersection.
        let cam1_images = if monocular {
            Vec::new()
        } else {
            timing.measure(TimingBucket::DatasetCsvParsing, || {
                read_image_manifest(&cam1_dir.join("data.csv"), &cam1_dir)
            })?
        };
        let mut cam1_by_timestamp = BTreeMap::new();
        for image in cam1_images {
            if cam1_by_timestamp
                .insert(image.timestamp_ns, image.clone())
                .is_some()
            {
                return Err(EurocReaderError::DuplicateTimestamp {
                    path: cam1_dir.join("data.csv"),
                    timestamp_ns: image.timestamp_ns,
                });
            }
        }
        let cam0_manifest_count = cam0_images.len();
        if !monocular {
            cam0_images.retain(|image| cam1_by_timestamp.contains_key(&image.timestamp_ns));
        }
        if cam0_images.is_empty() {
            return Err(EurocReaderError::EmptyStereoIntersection {
                cam0: cam0_dir.join("data.csv"),
                cam1: cam1_dir.join("data.csv"),
            });
        }
        let imu_samples = timing.measure(TimingBucket::DatasetCsvParsing, || {
            read_imu_csv(&imu0_dir.join("data.csv"))
        })?;

        let calibration_path = calibration_path.as_ref();
        let calibration = timing.measure(TimingBucket::DatasetCalibration, || {
            let calibration_json =
                fs::read_to_string(calibration_path).map_err(|source| EurocReaderError::Io {
                    path: calibration_path.to_path_buf(),
                    source,
                })?;
            BasaltCalibration::from_json_str(&calibration_json).map_err(EurocReaderError::from)
        })?;
        let config_path = config_path.as_ref();
        let config = timing.measure(TimingBucket::DatasetConfig, || {
            let config_json =
                fs::read_to_string(config_path).map_err(|source| EurocReaderError::Io {
                    path: config_path.to_path_buf(),
                    source,
                })?;
            BasaltConfig::from_json(&config_json).map_err(EurocReaderError::from)
        })?;

        let calibration = if monocular {
            calibration.retain_cameras(1)?
        } else {
            calibration
        };

        Ok(Self {
            root,
            cam0_images,
            cam0_manifest_count,
            cam1_by_timestamp,
            imu_samples,
            calibration,
            config,
            monocular,
        })
    }

    /// True when opened with [`Self::open_monocular`] (cam0 + IMU only).
    pub const fn is_monocular(&self) -> bool {
        self.monocular
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn calibration(&self) -> &BasaltCalibration {
        &self.calibration
    }

    pub const fn config(&self) -> &BasaltConfig {
        &self.config
    }

    pub fn frame_count(&self) -> usize {
        self.cam0_images.len()
    }

    /// Number of rows in the original cam0 manifest, before requiring the
    /// stereo timestamp intersection used by upstream Basalt optical flow.
    pub const fn cam0_manifest_count(&self) -> usize {
        self.cam0_manifest_count
    }

    pub fn cam0_images(&self) -> &[EurocImageEntry] {
        &self.cam0_images
    }

    pub fn imu_samples(&self) -> &[ImuSample] {
        &self.imu_samples
    }

    pub fn cam1_timestamp_count(&self) -> usize {
        self.cam1_by_timestamp.len()
    }

    /// Decodes one cam0/cam1 pair and selects its half-open IMU interval.
    pub fn frame(&self, index: usize) -> Result<EurocSensorFrame, EurocReaderError> {
        self.frame_impl(index, None)
    }

    /// Decodes one stereo pair while collecting per-image decode and
    /// raw-u16 conversion timing.  All pixel expressions and interval
    /// selection are shared with [`Self::frame`].
    pub fn frame_with_timing(
        &self,
        index: usize,
        timing: &mut TimingBreakdown,
    ) -> Result<EurocSensorFrame, EurocReaderError> {
        self.frame_impl(index, Some(timing))
    }

    fn frame_impl(
        &self,
        index: usize,
        mut timing: Option<&mut TimingBreakdown>,
    ) -> Result<EurocSensorFrame, EurocReaderError> {
        let cam0 = self
            .cam0_images
            .get(index)
            .ok_or(EurocReaderError::FrameIndex { index })?;
        let previous_timestamp = index
            .checked_sub(1)
            .and_then(|previous| self.cam0_images.get(previous))
            .map(|entry| entry.timestamp_ns);

        let cam0_image = match timing.as_deref_mut() {
            Some(timing) => read_raw_u16_png_timed(&cam0.path, timing)?,
            None => read_raw_u16_png(&cam0.path)?,
        };
        let cam1_entry = self.cam1_by_timestamp.get(&cam0.timestamp_ns);
        let (cam1_image, cam1_path) = match cam1_entry {
            Some(entry) => (
                Some(match timing.as_deref_mut() {
                    Some(timing) => read_raw_u16_png_timed(&entry.path, timing)?,
                    None => read_raw_u16_png(&entry.path)?,
                }),
                Some(entry.path.clone()),
            ),
            None => (None, None),
        };

        let imu = self
            .imu_samples
            .iter()
            .filter(|sample| {
                previous_timestamp.is_none_or(|previous| sample.timestamp_ns > previous)
                    && sample.timestamp_ns <= cam0.timestamp_ns
            })
            .copied()
            .collect();

        Ok(EurocSensorFrame {
            frame_id: index as FrameId,
            timestamp_ns: cam0.timestamp_ns,
            cam0: cam0_image,
            cam1: cam1_image,
            cam0_path: cam0.path.clone(),
            cam1_path,
            imu,
            initialization_imu: (index == 0)
                .then(|| {
                    self.imu_samples
                        .iter()
                        .find(|sample| sample.timestamp_ns >= cam0.timestamp_ns)
                        .copied()
                })
                .flatten(),
        })
    }
}

fn read_image_manifest(
    path: &Path,
    base_dir: &Path,
) -> Result<Vec<EurocImageEntry>, EurocReaderError> {
    let text = fs::read_to_string(path).map_err(|source| EurocReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut entries = Vec::new();
    let mut previous_timestamp = None;
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = trimmed.split(',').map(str::trim).collect();
        if fields.len() < 2 {
            return Err(EurocReaderError::Csv {
                path: path.to_path_buf(),
                line: line_index + 1,
                message: "expected timestamp,filename".into(),
            });
        }
        let timestamp_ns = parse_i64(path, line_index + 1, fields[0])?;
        if previous_timestamp.is_some_and(|previous| timestamp_ns <= previous) {
            return Err(EurocReaderError::NonMonotonicTimestamp {
                path: path.to_path_buf(),
                line: line_index + 1,
                timestamp_ns,
            });
        }
        let filename = fields[1].to_owned();
        if filename.is_empty() {
            return Err(EurocReaderError::Csv {
                path: path.to_path_buf(),
                line: line_index + 1,
                message: "empty image filename".into(),
            });
        }
        validate_image_filename(&path, line_index + 1, &filename)?;
        entries.push(EurocImageEntry {
            timestamp_ns,
            filename: filename.clone(),
            path: base_dir.join("data").join(filename),
        });
        previous_timestamp = Some(timestamp_ns);
    }
    Ok(entries)
}

fn validate_image_filename(
    path: &Path,
    line: usize,
    filename: &str,
) -> Result<(), EurocReaderError> {
    let candidate = Path::new(filename);
    let mut components = candidate.components();
    let is_single_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    let has_path_syntax = filename
        .chars()
        .any(|character| matches!(character, '/' | '\\' | ':' | '\0'));
    if candidate.is_absolute() || !is_single_normal_component || has_path_syntax {
        return Err(EurocReaderError::Csv {
            path: path.to_path_buf(),
            line,
            message: "image filename must be one safe relative path component".into(),
        });
    }
    Ok(())
}

fn read_imu_csv(path: &Path) -> Result<Vec<ImuSample>, EurocReaderError> {
    let text = fs::read_to_string(path).map_err(|source| EurocReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut samples = Vec::new();
    let mut previous_timestamp = None;
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = trimmed.split(',').map(str::trim).collect();
        if fields.len() < 7 {
            return Err(EurocReaderError::Csv {
                path: path.to_path_buf(),
                line: line_index + 1,
                message: "expected timestamp, gyro xyz, accel xyz".into(),
            });
        }
        let timestamp_ns = parse_i64(path, line_index + 1, fields[0])?;
        if previous_timestamp.is_some_and(|previous| timestamp_ns <= previous) {
            return Err(EurocReaderError::NonMonotonicTimestamp {
                path: path.to_path_buf(),
                line: line_index + 1,
                timestamp_ns,
            });
        }
        let values = fields[1..7]
            .iter()
            .map(|field| parse_f64(path, line_index + 1, field))
            .collect::<Result<Vec<_>, _>>()?;
        let gyro = Vector3::new(values[0], values[1], values[2]);
        let accel = Vector3::new(values[3], values[4], values[5]);
        if !gyro
            .iter()
            .chain(accel.iter())
            .all(|value| value.is_finite())
        {
            return Err(EurocReaderError::Csv {
                path: path.to_path_buf(),
                line: line_index + 1,
                message: "IMU sample contains a non-finite value".into(),
            });
        }
        samples.push(ImuSample::new(timestamp_ns, gyro, accel));
        previous_timestamp = Some(timestamp_ns);
    }
    if samples.len() < 2 {
        return Err(EurocReaderError::InsufficientImuSamples(path.to_path_buf()));
    }
    Ok(samples)
}

fn parse_i64(path: &Path, line: usize, value: &str) -> Result<i64, EurocReaderError> {
    value.parse().map_err(|source| EurocReaderError::Csv {
        path: path.to_path_buf(),
        line,
        message: format!("invalid integer `{value}`: {source}"),
    })
}

fn parse_f64(path: &Path, line: usize, value: &str) -> Result<f64, EurocReaderError> {
    value.parse().map_err(|source| EurocReaderError::Csv {
        path: path.to_path_buf(),
        line,
        message: format!("invalid float `{value}`: {source}"),
    })
}

fn read_raw_u16_png(path: &Path) -> Result<RawU16Image, EurocReaderError> {
    let dynamic = image::open(path).map_err(|source| EurocReaderError::Image {
        path: path.to_path_buf(),
        source,
    })?;
    dynamic_to_raw_u16(path, dynamic)
}

fn read_raw_u16_png_timed(
    path: &Path,
    timing: &mut TimingBreakdown,
) -> Result<RawU16Image, EurocReaderError> {
    let dynamic = timing.measure(TimingBucket::DatasetPngOpenDecode, || {
        image::open(path).map_err(|source| EurocReaderError::Image {
            path: path.to_path_buf(),
            source,
        })
    })?;
    timing.measure(TimingBucket::DatasetRawU16Conversion, || {
        dynamic_to_raw_u16(path, dynamic)
    })
}

fn dynamic_to_raw_u16(path: &Path, dynamic: DynamicImage) -> Result<RawU16Image, EurocReaderError> {
    let color = format!("{:?}", dynamic.color());
    let (width, height) = (dynamic.width() as usize, dynamic.height() as usize);
    let pixels = match dynamic {
        DynamicImage::ImageLuma8(image) => image
            .into_raw()
            .into_iter()
            .map(|value| u16::from(value) << 8)
            .collect(),
        DynamicImage::ImageLuma16(image) => image.into_raw(),
        DynamicImage::ImageRgb8(image) => image
            .into_raw()
            .chunks_exact(3)
            .map(|pixel| u16::from(pixel[0]) << 8)
            .collect(),
        DynamicImage::ImageRgb16(image) => image
            .into_raw()
            .chunks_exact(3)
            .map(|pixel| pixel[0])
            .collect(),
        DynamicImage::ImageLumaA8(image) => image
            .into_raw()
            .chunks_exact(2)
            .map(|pixel| u16::from(pixel[0]) << 8)
            .collect(),
        DynamicImage::ImageLumaA16(image) => image
            .into_raw()
            .chunks_exact(2)
            .map(|pixel| pixel[0])
            .collect(),
        DynamicImage::ImageRgba8(image) => image
            .into_raw()
            .chunks_exact(4)
            .map(|pixel| u16::from(pixel[0]) << 8)
            .collect(),
        DynamicImage::ImageRgba16(image) => image
            .into_raw()
            .chunks_exact(4)
            .map(|pixel| pixel[0])
            .collect(),
        _ => {
            return Err(EurocReaderError::UnsupportedPixelFormat {
                path: path.to_path_buf(),
                color,
            })
        }
    };
    RawU16Image::new(width, height, pixels).map_err(EurocReaderError::RawImage)
}

/// Errors raised by the sensor-only reader.
#[derive(Debug, Error)]
pub enum EurocReaderError {
    #[error("EuRoC dataset directory does not exist: {0}")]
    MissingDirectory(PathBuf),
    #[error("missing or unreadable `{path}`: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("empty EuRoC image manifest: {0}")]
    EmptyManifest(PathBuf),
    #[error("cam0/cam1 manifests have no common timestamp (`{cam0}`, `{cam1}`)")]
    EmptyStereoIntersection { cam0: PathBuf, cam1: PathBuf },
    #[error("invalid CSV `{path}` line {line}: {message}")]
    Csv {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("non-monotonic timestamp in `{path}` line {line}: {timestamp_ns}")]
    NonMonotonicTimestamp {
        path: PathBuf,
        line: usize,
        timestamp_ns: i64,
    },
    #[error("fewer than two IMU samples in `{0}`")]
    InsufficientImuSamples(PathBuf),
    #[error("duplicate timestamp {timestamp_ns} in `{path}`")]
    DuplicateTimestamp { path: PathBuf, timestamp_ns: i64 },
    #[error("EuRoC frame index is out of range: {index}")]
    FrameIndex { index: usize },
    #[error("failed to read PNG `{path}`: {source}")]
    Image {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("unsupported PNG pixel format for `{path}`: {color}")]
    UnsupportedPixelFormat { path: PathBuf, color: String },
    #[error("raw image contract failed: {0}")]
    RawImage(#[from] RawImageError),
    #[error("Basalt calibration error: {0}")]
    Calibration(#[from] CalibrationError),
    #[error("Basalt config error: {0}")]
    Config(#[from] ConfigError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "visloc_basalt_euroc_reader_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_file(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, text).unwrap();
    }

    fn make_dataset() -> (PathBuf, PathBuf, PathBuf) {
        let root = temp_dir();
        for camera in ["cam0", "cam1"] {
            let directory = root.join("mav0").join(camera);
            fs::create_dir_all(directory.join("data")).unwrap();
            write_file(
                &directory.join("data.csv"),
                "#timestamp,filename\n100,100.png\n200,200.png\n",
            );
            for (timestamp, pixels) in [(100, vec![0, 255, 64, 128]), (200, vec![1, 2, 3, 4])] {
                GrayImage::from_raw(2, 2, pixels)
                    .unwrap()
                    .save(directory.join("data").join(format!("{timestamp}.png")))
                    .unwrap();
            }
        }
        write_file(
            &root.join("mav0/imu0/data.csv"),
            "#timestamp,gx,gy,gz,ax,ay,az\n50,0,0,0,0,0,9.8\n100,0,0,0,0,0,9.8\n150,0,0,0,0,0,9.8\n200,0,0,0,0,0,9.8\n",
        );
        // A malformed file in the GT location proves that the sensor-only
        // reader never opens or parses that tree.
        write_file(
            &root.join("mav0/state_groundtruth_estimate0/data.csv"),
            "this is intentionally not a valid trajectory\n",
        );
        let calibration = root.join("calib.json");
        write_file(
            &calibration,
            include_str!("../tests/fixtures/euroc_ds_calib_minimal.json"),
        );
        let config = root.join("config.json");
        write_file(
            &config,
            include_str!("../../../configs/basalt/euroc_config.json"),
        );
        (root, calibration, config)
    }

    #[test]
    fn sensor_reader_ignores_gt_and_preserves_u16_container_contract() {
        let (root, calibration, config) = make_dataset();
        let dataset = EurocSensorDataset::open(&root, calibration, config).unwrap();
        assert_eq!(dataset.frame_count(), 2);
        assert_eq!(dataset.cam1_timestamp_count(), 2);
        assert_eq!(dataset.imu_samples().len(), 4);
        let first = dataset.frame(0).unwrap();
        assert_eq!(first.frame_id, 0);
        assert_eq!(first.imu.len(), 2);
        assert_eq!(first.initialization_imu.unwrap().timestamp_ns, 100);
        assert_eq!(first.cam0.pixel(0, 0), Some(0));
        assert_eq!(first.cam0.pixel(1, 0), Some(65_280));
        assert_eq!(first.cam1.as_ref().unwrap().pixel(0, 0), Some(0));
        let second = dataset.frame(1).unwrap();
        assert!(second.initialization_imu.is_none());
        assert_eq!(second.imu.len(), 2);
        assert_eq!(second.imu[0].timestamp_ns, 150);
        assert_eq!(second.imu[1].timestamp_ns, 200);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn first_frame_exposes_future_imu_only_for_initialization() {
        let (root, calibration, config) = make_dataset();
        write_file(
            &root.join("mav0/imu0/data.csv"),
            "#timestamp,gx,gy,gz,ax,ay,az\n125,0,0,0,0,9.8,0\n150,0,0,0,0,9.8,0\n200,0,0,0,0,9.8,0\n",
        );
        let dataset = EurocSensorDataset::open(&root, calibration, config).unwrap();

        let first = dataset.frame(0).unwrap();
        assert!(first.imu.is_empty());
        assert_eq!(first.initialization_imu.unwrap().timestamp_ns, 125);

        let second = dataset.frame(1).unwrap();
        assert!(second.initialization_imu.is_none());
        assert_eq!(
            second
                .imu
                .iter()
                .map(|sample| sample.timestamp_ns)
                .collect::<Vec<_>>(),
            vec![125, 150, 200]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reader_uses_stereo_timestamp_intersection_like_upstream_optical_flow() {
        let (root, calibration, config) = make_dataset();
        write_file(
            &root.join("mav0/cam1/data.csv"),
            "#timestamp,filename\n200,200.png\n",
        );
        let dataset = EurocSensorDataset::open(&root, calibration, config).unwrap();
        assert_eq!(dataset.cam0_manifest_count(), 2);
        assert_eq!(dataset.frame_count(), 1);

        let first = dataset.frame(0).unwrap();
        assert_eq!(first.timestamp_ns, 200);
        assert_eq!(first.initialization_imu.unwrap().timestamp_ns, 200);
        assert!(first.cam1.is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn monocular_reader_ignores_cam1_and_keeps_every_cam0_frame() {
        let (root, calibration, config) = make_dataset();
        // A cam1 manifest that would shrink the stereo intersection, then
        // no cam1 tree at all: neither may affect the monocular stream.
        write_file(
            &root.join("mav0/cam1/data.csv"),
            "#timestamp,filename\n200,200.png\n",
        );
        let stereo = EurocSensorDataset::open(&root, &calibration, &config).unwrap();
        assert!(!stereo.is_monocular());
        assert_eq!(stereo.frame_count(), 1);
        fs::remove_dir_all(root.join("mav0/cam1")).unwrap();

        let dataset = EurocSensorDataset::open_monocular(&root, &calibration, &config).unwrap();
        assert!(dataset.is_monocular());
        assert_eq!(dataset.frame_count(), 2);
        assert_eq!(dataset.cam0_manifest_count(), 2);
        assert_eq!(dataset.cam1_timestamp_count(), 0);
        assert_eq!(dataset.calibration().cameras.len(), 1);
        assert_eq!(dataset.calibration().t_imu_cam.len(), 1);
        for index in 0..2 {
            let frame = dataset.frame(index).unwrap();
            assert!(frame.cam1.is_none());
            assert!(frame.cam1_path.is_none());
        }
        let first = dataset.frame(0).unwrap();
        assert_eq!(first.timestamp_ns, 100);
        assert_eq!(first.initialization_imu.unwrap().timestamp_ns, 100);
        assert_eq!(first.cam0.pixel(1, 0), Some(65_280));
        assert!(EurocSensorDataset::open(&root, &calibration, &config).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_sensor_manifest_is_rejected_without_gt_fallback() {
        let (root, calibration, config) = make_dataset();
        write_file(
            &root.join("mav0/cam0/data.csv"),
            "#timestamp,filename\n200,200.png\n100,100.png\n",
        );
        let error = EurocSensorDataset::open(root.clone(), calibration, config).unwrap_err();
        assert!(matches!(
            error,
            EurocReaderError::NonMonotonicTimestamp { .. }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sensor_manifest_rejects_absolute_traversal_and_separator_filenames() {
        for filename in [
            "../../state_groundtruth_estimate0/data/pose.csv",
            "nested/frame.png",
            "C:\\outside.png",
        ] {
            let (root, calibration, config) = make_dataset();
            write_file(
                &root.join("mav0/cam0/data.csv"),
                &format!("#timestamp,filename\n100,{filename}\n"),
            );
            let error = EurocSensorDataset::open(root.clone(), calibration, config).unwrap_err();
            assert!(matches!(
                error,
                EurocReaderError::Csv { message, .. }
                    if message.contains("safe relative path component")
            ));
            fs::remove_dir_all(root).unwrap();
        }
    }
}
