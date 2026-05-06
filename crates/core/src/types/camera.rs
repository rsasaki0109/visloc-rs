use nalgebra::{Point2, Point3};

pub type CameraId = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraModel {
    Pinhole,
    SimplePinhole,
    SimpleRadial,
    Radial,
    OpenCv,
    Unknown(String),
}

impl CameraModel {
    pub fn from_colmap_name(name: &str) -> Self {
        match name {
            "PINHOLE" => Self::Pinhole,
            "SIMPLE_PINHOLE" => Self::SimplePinhole,
            "SIMPLE_RADIAL" => Self::SimpleRadial,
            "RADIAL" => Self::Radial,
            "OPENCV" => Self::OpenCv,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub id: CameraId,
    pub model: CameraModel,
    pub width: u32,
    pub height: u32,
    pub params: Vec<f64>,
}

impl Camera {
    pub fn pinhole(
        id: CameraId,
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
    ) -> Self {
        Self {
            id,
            model: CameraModel::Pinhole,
            width,
            height,
            params: vec![fx, fy, cx, cy],
        }
    }

    pub fn intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        match self.model {
            CameraModel::Pinhole | CameraModel::OpenCv => Some((
                *self.params.first()?,
                *self.params.get(1)?,
                *self.params.get(2)?,
                *self.params.get(3)?,
            )),
            CameraModel::SimplePinhole | CameraModel::SimpleRadial | CameraModel::Radial => {
                let f = *self.params.first()?;
                Some((f, f, *self.params.get(1)?, *self.params.get(2)?))
            }
            CameraModel::Unknown(_) => None,
        }
    }

    pub fn normalize_pixel(&self, point: &Point2<f64>) -> Option<Point2<f64>> {
        let (fx, fy, cx, cy) = self.intrinsics()?;
        Some(Point2::new((point.x - cx) / fx, (point.y - cy) / fy))
    }

    pub fn project(&self, point_camera: &Point3<f64>) -> Option<Point2<f64>> {
        if point_camera.z <= 0.0 {
            return None;
        }
        let (fx, fy, cx, cy) = self.intrinsics()?;
        Some(Point2::new(
            fx * point_camera.x / point_camera.z + cx,
            fy * point_camera.y / point_camera.z + cy,
        ))
    }
}
