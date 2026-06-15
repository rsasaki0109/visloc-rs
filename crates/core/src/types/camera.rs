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

    /// Construct a pinhole camera with two trailing radial-distortion
    /// coefficients `(k1, k2)`. The intrinsics layout stays `[fx, fy, cx, cy]`
    /// (so [`Self::intrinsics`] is unchanged); the distortion lives in the two
    /// extra `params` slots and is read back by [`Self::radial_distortion`].
    #[allow(clippy::too_many_arguments)]
    pub fn pinhole_radial(
        id: CameraId,
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
    ) -> Self {
        Self {
            id,
            model: CameraModel::Pinhole,
            width,
            height,
            params: vec![fx, fy, cx, cy, k1, k2],
        }
    }

    /// Radial-distortion coefficients `(k1, k2)` carried alongside the
    /// pinhole intrinsics, if any. A plain 4-parameter pinhole returns `None`
    /// (distortion-free); `Pinhole` / `OpenCv` read the optional trailing
    /// `[k1, k2]`, while COLMAP `SimpleRadial` / `Radial` read their native
    /// `[f, cx, cy, k1(, k2)]` layout.
    pub fn radial_distortion(&self) -> Option<(f64, f64)> {
        match self.model {
            CameraModel::Pinhole | CameraModel::OpenCv => self
                .params
                .get(4)
                .map(|&k1| (k1, self.params.get(5).copied().unwrap_or(0.0))),
            CameraModel::SimpleRadial => self.params.get(3).map(|&k1| (k1, 0.0)),
            CameraModel::Radial => self
                .params
                .get(3)
                .map(|&k1| (k1, self.params.get(4).copied().unwrap_or(0.0))),
            CameraModel::SimplePinhole | CameraModel::Unknown(_) => None,
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

    /// Back-project a pixel to a normalized (undistorted) bearing `(x, y, 1)`.
    /// When the camera carries radial distortion the pixel is undistorted first
    /// (fixed-point inverse of `1 + k1·r² + k2·r⁴`), so the returned coordinates
    /// match an ideal pinhole — what the geometric front-end (essential matrix,
    /// PnP, triangulation) expects. A distortion-free camera is the unchanged
    /// linear back-projection.
    pub fn normalize_pixel(&self, point: &Point2<f64>) -> Option<Point2<f64>> {
        let (fx, fy, cx, cy) = self.intrinsics()?;
        let xd = (point.x - cx) / fx;
        let yd = (point.y - cy) / fy;
        match self.radial_distortion() {
            Some((k1, k2)) if k1 != 0.0 || k2 != 0.0 => Some(undistort_radial(xd, yd, k1, k2)),
            _ => Some(Point2::new(xd, yd)),
        }
    }

    pub fn project(&self, point_camera: &Point3<f64>) -> Option<Point2<f64>> {
        if point_camera.z <= 0.0 {
            return None;
        }
        let (fx, fy, cx, cy) = self.intrinsics()?;
        let mut x = point_camera.x / point_camera.z;
        let mut y = point_camera.y / point_camera.z;
        if let Some((k1, k2)) = self.radial_distortion() {
            if k1 != 0.0 || k2 != 0.0 {
                let r2 = x * x + y * y;
                let d = 1.0 + k1 * r2 + k2 * r2 * r2;
                x *= d;
                y *= d;
            }
        }
        Some(Point2::new(fx * x + cx, fy * y + cy))
    }
}

/// Fixed-point inverse of the radial-distortion map `(x, y) ↦ (x, y)·(1 + k1·r²
/// + k2·r⁴)` on normalized coordinates. Converges in well under 20 steps for
/// realistic lens distortion; mirrors `visloc_vision::distortion` (kept here so
/// `visloc-core` stays dependency-free).
fn undistort_radial(xd: f64, yd: f64, k1: f64, k2: f64) -> Point2<f64> {
    let (mut x, mut y) = (xd, yd);
    for _ in 0..20 {
        let r2 = x * x + y * y;
        let d = 1.0 + k1 * r2 + k2 * r2 * r2;
        let nx = xd / d;
        let ny = yd / d;
        if (nx - x).abs() + (ny - y).abs() < 1.0e-12 {
            return Point2::new(nx, ny);
        }
        x = nx;
        y = ny;
    }
    Point2::new(x, y)
}
