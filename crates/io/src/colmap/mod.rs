use nalgebra::{Point2, Point3, Quaternion, UnitQuaternion, Vector3};
use std::fs;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use thiserror::Error;
use visloc_core::geometry::Pose;
use visloc_core::types::{
    Camera, CameraModel, Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation,
    VisualMap, VisualMapValidationReport,
};
use visloc_localization::{DescriptorProvider, MapProvider};

use crate::descriptors::{read_landmark_descriptors_txt, DescriptorStoreError};

#[derive(Debug, Error)]
pub enum ColmapError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer parse error: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("float parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("invalid COLMAP line in {file}: {line}")]
    InvalidLine { file: &'static str, line: String },
    #[error("invalid COLMAP binary in {file}: {message}")]
    InvalidBinary { file: &'static str, message: String },
}

#[derive(Debug, Error)]
pub enum ColmapMapProviderError {
    #[error("COLMAP error: {0}")]
    Colmap(#[from] ColmapError),
    #[error("descriptor store error: {0}")]
    DescriptorStore(#[from] DescriptorStoreError),
    #[error("COLMAP map validation failed: {0:?}")]
    InvalidMap(VisualMapValidationReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapMapProvider {
    pub map: VisualMap,
    pub descriptor_store: Option<LandmarkDescriptorStore>,
}

impl ColmapMapProvider {
    pub fn from_text_model_dir(path: impl AsRef<Path>) -> Result<Self, ColmapMapProviderError> {
        Ok(Self {
            map: read_colmap_text_model(path)?,
            descriptor_store: None,
        })
    }

    pub fn from_binary_model_dir(path: impl AsRef<Path>) -> Result<Self, ColmapMapProviderError> {
        Ok(Self {
            map: read_colmap_binary_model(path)?,
            descriptor_store: None,
        })
    }

    pub fn from_text_model_dir_validated(
        path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        let provider = Self::from_text_model_dir(path)?;
        provider
            .validate_map()
            .into_result()
            .map_err(ColmapMapProviderError::InvalidMap)?;
        Ok(provider)
    }

    pub fn from_binary_model_dir_validated(
        path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        let provider = Self::from_binary_model_dir(path)?;
        provider
            .validate_map()
            .into_result()
            .map_err(ColmapMapProviderError::InvalidMap)?;
        Ok(provider)
    }

    pub fn from_text_model_dir_with_descriptors(
        model_path: impl AsRef<Path>,
        descriptor_path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        Ok(Self {
            map: read_colmap_text_model(model_path)?,
            descriptor_store: Some(read_landmark_descriptors_txt(descriptor_path)?),
        })
    }

    pub fn from_text_model_dir_with_descriptors_validated(
        model_path: impl AsRef<Path>,
        descriptor_path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        let provider = Self::from_text_model_dir_with_descriptors(model_path, descriptor_path)?;
        provider
            .validate_for_localization()
            .into_result()
            .map_err(ColmapMapProviderError::InvalidMap)?;
        Ok(provider)
    }

    pub fn validate_map(&self) -> VisualMapValidationReport {
        self.map.validate()
    }

    pub fn validate_for_localization(&self) -> VisualMapValidationReport {
        self.map
            .validate_with_descriptors(self.descriptor_store.as_ref())
    }
}

impl MapProvider for ColmapMapProvider {
    fn visual_map(&self) -> &VisualMap {
        &self.map
    }
}

impl DescriptorProvider for ColmapMapProvider {
    fn landmark_descriptor_store(&self) -> Option<&LandmarkDescriptorStore> {
        self.descriptor_store.as_ref()
    }
}

pub fn read_colmap_text_model(path: impl AsRef<Path>) -> Result<VisualMap, ColmapError> {
    let path = path.as_ref();
    let mut map = VisualMap::new();

    let cameras = parse_cameras_txt(&fs::read_to_string(path.join("cameras.txt"))?)?;
    map.cameras = cameras
        .into_iter()
        .map(|camera| (camera.id, camera))
        .collect();

    let landmarks = parse_points3d_txt(&fs::read_to_string(path.join("points3D.txt"))?)?;
    map.landmarks = landmarks
        .into_iter()
        .map(|landmark| (landmark.id, landmark))
        .collect();

    let keyframes = parse_images_txt(&fs::read_to_string(path.join("images.txt"))?)?;
    map.keyframes = keyframes
        .into_iter()
        .map(|keyframe| (keyframe.frame.id, keyframe))
        .collect();

    Ok(map)
}

pub fn read_colmap_binary_model(path: impl AsRef<Path>) -> Result<VisualMap, ColmapError> {
    let path = path.as_ref();
    let mut map = VisualMap::new();

    let cameras = parse_cameras_bin(&fs::read(path.join("cameras.bin"))?)?;
    map.cameras = cameras
        .into_iter()
        .map(|camera| (camera.id, camera))
        .collect();

    let landmarks = parse_points3d_bin(&fs::read(path.join("points3D.bin"))?)?;
    map.landmarks = landmarks
        .into_iter()
        .map(|landmark| (landmark.id, landmark))
        .collect();

    let keyframes = parse_images_bin(&fs::read(path.join("images.bin"))?)?;
    map.keyframes = keyframes
        .into_iter()
        .map(|keyframe| (keyframe.frame.id, keyframe))
        .collect();

    Ok(map)
}

pub fn parse_cameras_txt(contents: &str) -> Result<Vec<Camera>, ColmapError> {
    let mut cameras = Vec::new();
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 5 {
            return Err(ColmapError::InvalidLine {
                file: "cameras.txt",
                line: line.to_owned(),
            });
        }
        let id = tokens[0].parse()?;
        let model = CameraModel::from_colmap_name(tokens[1]);
        let width = tokens[2].parse()?;
        let height = tokens[3].parse()?;
        let params = tokens[4..]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        cameras.push(Camera {
            id,
            model,
            width,
            height,
            params,
        });
    }
    Ok(cameras)
}

pub fn parse_points3d_txt(contents: &str) -> Result<Vec<Landmark>, ColmapError> {
    let mut landmarks = Vec::new();
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 8 {
            return Err(ColmapError::InvalidLine {
                file: "points3D.txt",
                line: line.to_owned(),
            });
        }
        landmarks.push(Landmark::new(
            tokens[0].parse()?,
            Point3::new(tokens[1].parse()?, tokens[2].parse()?, tokens[3].parse()?),
        ));
    }
    Ok(landmarks)
}

pub fn parse_images_txt(contents: &str) -> Result<Vec<Keyframe>, ColmapError> {
    let mut keyframes = Vec::new();
    let mut lines = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'));

    while let Some(header) = lines.next() {
        if header.is_empty() {
            continue;
        }

        let tokens = header.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 10 {
            return Err(ColmapError::InvalidLine {
                file: "images.txt",
                line: header.to_owned(),
            });
        }

        let frame_id = tokens[0].parse()?;
        let qw = tokens[1].parse()?;
        let qx = tokens[2].parse()?;
        let qy = tokens[3].parse()?;
        let qz = tokens[4].parse()?;
        let tx = tokens[5].parse()?;
        let ty = tokens[6].parse()?;
        let tz = tokens[7].parse()?;
        let camera_id = tokens[8].parse()?;
        let points_line = lines.next().unwrap_or("");

        let mut frame = Frame::new(frame_id, camera_id);
        frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
            Vector3::new(tx, ty, tz),
        ));

        let point_tokens = points_line.split_whitespace().collect::<Vec<_>>();
        if point_tokens.len() % 3 != 0 {
            return Err(ColmapError::InvalidLine {
                file: "images.txt",
                line: points_line.to_owned(),
            });
        }

        let mut observations = Vec::new();
        for (keypoint_index, chunk) in point_tokens.chunks(3).enumerate() {
            let xy = Point2::new(chunk[0].parse()?, chunk[1].parse()?);
            frame.keypoints.push(xy);

            let point_id = chunk[2].parse::<i64>()?;
            if point_id >= 0 {
                observations.push(Observation {
                    frame_id,
                    landmark_id: point_id as u64,
                    keypoint_index,
                    xy,
                });
            }
        }

        keyframes.push(Keyframe {
            frame,
            observations,
        });
    }

    Ok(keyframes)
}

pub fn parse_cameras_bin(contents: &[u8]) -> Result<Vec<Camera>, ColmapError> {
    let mut reader = BinaryReader::new("cameras.bin", contents);
    let camera_count = reader.read_u64()? as usize;
    let mut cameras = Vec::with_capacity(camera_count);

    for _ in 0..camera_count {
        let id = reader.read_u32()? as u64;
        let model_id = reader.read_i32()?;
        let width = u32::try_from(reader.read_u64()?).map_err(|_| ColmapError::InvalidBinary {
            file: "cameras.bin",
            message: "camera width does not fit in u32".to_owned(),
        })?;
        let height = u32::try_from(reader.read_u64()?).map_err(|_| ColmapError::InvalidBinary {
            file: "cameras.bin",
            message: "camera height does not fit in u32".to_owned(),
        })?;
        let (model, parameter_count) = camera_model_from_colmap_id(model_id)?;
        let mut params = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count {
            params.push(reader.read_f64()?);
        }
        cameras.push(Camera {
            id,
            model,
            width,
            height,
            params,
        });
    }

    reader.finish()?;
    Ok(cameras)
}

pub fn parse_images_bin(contents: &[u8]) -> Result<Vec<Keyframe>, ColmapError> {
    let mut reader = BinaryReader::new("images.bin", contents);
    let image_count = reader.read_u64()? as usize;
    let mut keyframes = Vec::with_capacity(image_count);

    for _ in 0..image_count {
        let frame_id = reader.read_u32()? as u64;
        let qw = reader.read_f64()?;
        let qx = reader.read_f64()?;
        let qy = reader.read_f64()?;
        let qz = reader.read_f64()?;
        let tx = reader.read_f64()?;
        let ty = reader.read_f64()?;
        let tz = reader.read_f64()?;
        let camera_id = reader.read_u32()? as u64;
        reader.read_null_terminated_string()?;

        let mut frame = Frame::new(frame_id, camera_id);
        frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
            Vector3::new(tx, ty, tz),
        ));

        let point_count = reader.read_u64()? as usize;
        let mut observations = Vec::new();
        for keypoint_index in 0..point_count {
            let xy = Point2::new(reader.read_f64()?, reader.read_f64()?);
            frame.keypoints.push(xy);

            let point_id = reader.read_i64()?;
            if point_id >= 0 {
                observations.push(Observation {
                    frame_id,
                    landmark_id: point_id as u64,
                    keypoint_index,
                    xy,
                });
            }
        }

        keyframes.push(Keyframe {
            frame,
            observations,
        });
    }

    reader.finish()?;
    Ok(keyframes)
}

pub fn parse_points3d_bin(contents: &[u8]) -> Result<Vec<Landmark>, ColmapError> {
    let mut reader = BinaryReader::new("points3D.bin", contents);
    let point_count = reader.read_u64()? as usize;
    let mut landmarks = Vec::with_capacity(point_count);

    for _ in 0..point_count {
        let id = reader.read_u64()?;
        let position = Point3::new(reader.read_f64()?, reader.read_f64()?, reader.read_f64()?);
        reader.skip(3)?;
        let _error = reader.read_f64()?;
        let track_length = reader.read_u64()? as usize;
        for _ in 0..track_length {
            let _image_id = reader.read_u32()?;
            let _point2d_index = reader.read_u32()?;
        }
        landmarks.push(Landmark::new(id, position));
    }

    reader.finish()?;
    Ok(landmarks)
}

fn camera_model_from_colmap_id(model_id: i32) -> Result<(CameraModel, usize), ColmapError> {
    match model_id {
        0 => Ok((CameraModel::SimplePinhole, 3)),
        1 => Ok((CameraModel::Pinhole, 4)),
        2 => Ok((CameraModel::SimpleRadial, 4)),
        3 => Ok((CameraModel::Radial, 5)),
        4 => Ok((CameraModel::OpenCv, 8)),
        5 => Ok((CameraModel::Unknown("OPENCV_FISHEYE".to_owned()), 8)),
        6 => Ok((CameraModel::Unknown("FULL_OPENCV".to_owned()), 12)),
        7 => Ok((CameraModel::Unknown("FOV".to_owned()), 5)),
        8 => Ok((CameraModel::Unknown("SIMPLE_RADIAL_FISHEYE".to_owned()), 4)),
        9 => Ok((CameraModel::Unknown("RADIAL_FISHEYE".to_owned()), 5)),
        10 => Ok((CameraModel::Unknown("THIN_PRISM_FISHEYE".to_owned()), 12)),
        other => Err(ColmapError::InvalidBinary {
            file: "cameras.bin",
            message: format!("unsupported camera model id {other}"),
        }),
    }
}

struct BinaryReader<'a> {
    file: &'static str,
    contents: &'a [u8],
    offset: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(file: &'static str, contents: &'a [u8]) -> Self {
        Self {
            file,
            contents,
            offset: 0,
        }
    }

    fn read_u32(&mut self) -> Result<u32, ColmapError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, ColmapError> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, ColmapError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64, ColmapError> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64, ColmapError> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    fn read_null_terminated_string(&mut self) -> Result<String, ColmapError> {
        let start = self.offset;
        while self.offset < self.contents.len() && self.contents[self.offset] != 0 {
            self.offset += 1;
        }
        if self.offset >= self.contents.len() {
            return Err(ColmapError::InvalidBinary {
                file: self.file,
                message: "unterminated string".to_owned(),
            });
        }
        let name = std::str::from_utf8(&self.contents[start..self.offset])
            .map_err(|error| ColmapError::InvalidBinary {
                file: self.file,
                message: format!("invalid UTF-8 string: {error}"),
            })?
            .to_owned();
        self.offset += 1;
        Ok(name)
    }

    fn skip(&mut self, byte_count: usize) -> Result<(), ColmapError> {
        self.read_bytes(byte_count).map(|_| ())
    }

    fn finish(&self) -> Result<(), ColmapError> {
        if self.offset == self.contents.len() {
            Ok(())
        } else {
            Err(ColmapError::InvalidBinary {
                file: self.file,
                message: format!(
                    "{} trailing byte(s) after parsing",
                    self.contents.len() - self.offset
                ),
            })
        }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], ColmapError> {
        let bytes = self.read_bytes(N)?;
        let mut array = [0; N];
        array.copy_from_slice(bytes);
        Ok(array)
    }

    fn read_bytes(&mut self, byte_count: usize) -> Result<&'a [u8], ColmapError> {
        let end =
            self.offset
                .checked_add(byte_count)
                .ok_or_else(|| ColmapError::InvalidBinary {
                    file: self.file,
                    message: "reader offset overflow".to_owned(),
                })?;
        if end > self.contents.len() {
            return Err(ColmapError::InvalidBinary {
                file: self.file,
                message: format!("unexpected end of file at byte {}", self.offset),
            });
        }
        let bytes = &self.contents[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
}
