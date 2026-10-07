use nalgebra::{Point2, Point3, Vector3};

pub type CameraId = u64;

const EPS: f64 = 1e-12;

/// Camera projection models understood by the shared geometry front-end.
///
/// Pinhole-family models (`Pinhole`, `SimplePinhole`, `SimpleRadial`,
/// `Radial`, `OpenCv`) use `[fx, fy, cx, cy]` intrinsics plus optional
/// radial-tangential coefficients.  The fisheye-family models use their own
/// parameter layouts (documented on [`Camera::intrinsics`]) and are dispatched
/// to the matching `project`/`normalize_pixel` math.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraModel {
    Pinhole,
    SimplePinhole,
    SimpleRadial,
    Radial,
    OpenCv,
    /// OpenCV rational model: `[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]`.
    FullOpenCv,
    /// OpenCV fisheye / Kannala-Brandt equidistant: `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    OpenCvFisheye,
    /// Equidistant with one radial coefficient: `[f, cx, cy, k1]`.
    SimpleRadialFisheye,
    /// Equidistant with two radial coefficients: `[f, cx, cy, k1, k2]`.
    RadialFisheye,
    /// Devernay FOV model: `[fx, fy, cx, cy, omega]`.
    Fov,
    /// Double Sphere model: `[fx, fy, cx, cy, xi, alpha]`.
    DoubleSphere,
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
            "FULL_OPENCV" => Self::FullOpenCv,
            "OPENCV_FISHEYE" => Self::OpenCvFisheye,
            "SIMPLE_RADIAL_FISHEYE" => Self::SimpleRadialFisheye,
            "RADIAL_FISHEYE" => Self::RadialFisheye,
            "FOV" => Self::Fov,
            other => Self::Unknown(other.to_owned()),
        }
    }

    /// Canonical COLMAP model name, when this model has one.  Double Sphere is
    /// not a COLMAP model and returns `None`.
    pub fn colmap_name(&self) -> Option<&str> {
        Some(match self {
            Self::Pinhole => "PINHOLE",
            Self::SimplePinhole => "SIMPLE_PINHOLE",
            Self::SimpleRadial => "SIMPLE_RADIAL",
            Self::Radial => "RADIAL",
            Self::OpenCv => "OPENCV",
            Self::FullOpenCv => "FULL_OPENCV",
            Self::OpenCvFisheye => "OPENCV_FISHEYE",
            Self::SimpleRadialFisheye => "SIMPLE_RADIAL_FISHEYE",
            Self::RadialFisheye => "RADIAL_FISHEYE",
            Self::Fov => "FOV",
            Self::DoubleSphere => return None,
            Self::Unknown(name) => return Some(name.as_str()),
        })
    }

    /// Whether this model is a fisheye/wide-angle model whose pixel-to-ray map
    /// is not the pinhole `(x/z, y/z)` projection.
    pub const fn is_fisheye(&self) -> bool {
        matches!(
            self,
            Self::OpenCvFisheye | Self::SimpleRadialFisheye | Self::RadialFisheye | Self::Fov
        )
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

    /// Construct an OpenCV-fisheye (Kannala-Brandt equidistant) camera with
    /// `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    #[allow(clippy::too_many_arguments)]
    pub fn opencv_fisheye(
        id: CameraId,
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k: [f64; 4],
    ) -> Self {
        Self {
            id,
            model: CameraModel::OpenCvFisheye,
            width,
            height,
            params: vec![fx, fy, cx, cy, k[0], k[1], k[2], k[3]],
        }
    }

    /// Construct a Double Sphere camera with `[fx, fy, cx, cy, xi, alpha]`.
    #[allow(clippy::too_many_arguments)]
    pub fn double_sphere(
        id: CameraId,
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        xi: f64,
        alpha: f64,
    ) -> Self {
        Self {
            id,
            model: CameraModel::DoubleSphere,
            width,
            height,
            params: vec![fx, fy, cx, cy, xi, alpha],
        }
    }

    /// Radial-distortion coefficients `(k1, k2)` carried alongside the
    /// pinhole intrinsics, if any. A plain 4-parameter pinhole returns `None`
    /// (distortion-free); `Pinhole` / `OpenCv` read the optional trailing
    /// `[k1, k2]` (an `OpenCv` camera's `(p1, p2)` are reported separately by
    /// [`Self::tangential_distortion`]), while COLMAP `SimpleRadial` / `Radial` read their native
    /// `[f, cx, cy, k1(, k2)]` layout. The fisheye and Double Sphere models
    /// carry their own distortion and return `None` here.
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
            _ => None,
        }
    }

    /// Tangential-distortion coefficients `(p1, p2)` of a COLMAP `OPENCV`
    /// camera (`[fx, fy, cx, cy, k1, k2, p1, p2]`), or `None` when the model is
    /// not `OpenCv`, the slots are absent, or both are zero. When present,
    /// [`Self::project`] and [`Self::normalize_pixel`] apply the full
    /// Brown-Conrady model (radial + tangential); without them the radial-only
    /// path is used, unchanged.
    pub fn tangential_distortion(&self) -> Option<(f64, f64)> {
        if self.model != CameraModel::OpenCv {
            return None;
        }
        let p1 = self.params.get(6).copied().unwrap_or(0.0);
        let p2 = self.params.get(7).copied().unwrap_or(0.0);
        (p1 != 0.0 || p2 != 0.0).then_some((p1, p2))
    }

    pub fn intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        match self.model {
            CameraModel::Pinhole | CameraModel::OpenCv | CameraModel::FullOpenCv => Some((
                *self.params.first()?,
                *self.params.get(1)?,
                *self.params.get(2)?,
                *self.params.get(3)?,
            )),
            CameraModel::SimplePinhole
            | CameraModel::SimpleRadial
            | CameraModel::Radial
            | CameraModel::SimpleRadialFisheye
            | CameraModel::RadialFisheye => {
                let f = *self.params.first()?;
                Some((f, f, *self.params.get(1)?, *self.params.get(2)?))
            }
            CameraModel::OpenCvFisheye | CameraModel::Fov | CameraModel::DoubleSphere => Some((
                *self.params.first()?,
                *self.params.get(1)?,
                *self.params.get(2)?,
                *self.params.get(3)?,
            )),
            CameraModel::Unknown(_) => None,
        }
    }

    /// Back-project a pixel to a normalized (undistorted) bearing `(x, y, 1)`.
    ///
    /// Pinhole models with radial distortion are undistorted first
    /// (fixed-point inverse of `1 + k1·r² + k2·r⁴`), so the returned
    /// coordinates match an ideal pinhole — what the geometric front-end
    /// (essential matrix, PnP, triangulation) expects.  Fisheye/wide-angle
    /// models return the `(x, y, 1)` form of the recovered unit ray, which
    /// exists only while the ray still points forward (`z > 0`); use
    /// [`Self::unit_ray_from_pixel`] for the full field of view.
    pub fn normalize_pixel(&self, point: &Point2<f64>) -> Option<Point2<f64>> {
        if matches!(
            self.model,
            CameraModel::OpenCvFisheye
                | CameraModel::SimpleRadialFisheye
                | CameraModel::RadialFisheye
                | CameraModel::Fov
                | CameraModel::DoubleSphere
        ) {
            let ray = self.unit_ray_from_pixel(point)?;
            if ray.z <= EPS {
                return None;
            }
            return Some(Point2::new(ray.x / ray.z, ray.y / ray.z));
        }
        // `OPENCV` is `FULL_OPENCV` with k3..k6 = 0, so a camera with non-zero
        // (p1, p2) shares its iterative undistortion.
        if self.model == CameraModel::FullOpenCv || self.tangential_distortion().is_some() {
            let (fx, fy, cx, cy) = self.intrinsics()?;
            let xd = (point.x - cx) / fx;
            let yd = (point.y - cy) / fy;
            return Some(normalize_full_opencv(
                xd,
                yd,
                full_opencv_coeffs(&self.params),
            ));
        }
        let (fx, fy, cx, cy) = self.intrinsics()?;
        let xd = (point.x - cx) / fx;
        let yd = (point.y - cy) / fy;
        match self.radial_distortion() {
            Some((k1, k2)) if k1 != 0.0 || k2 != 0.0 => Some(undistort_radial(xd, yd, k1, k2)),
            _ => Some(Point2::new(xd, yd)),
        }
    }

    pub fn project(&self, point_camera: &Point3<f64>) -> Option<Point2<f64>> {
        if !point_camera.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        match self.model {
            CameraModel::DoubleSphere => {
                let [fx, fy, cx, cy, xi, alpha] = params6(&self.params)?;
                project_double_sphere(fx, fy, cx, cy, xi, alpha, point_camera)
            }
            CameraModel::OpenCvFisheye => {
                let [fx, fy, cx, cy] = params4(&self.params)?;
                let k = [
                    self.params.get(4).copied().unwrap_or(0.0),
                    self.params.get(5).copied().unwrap_or(0.0),
                    self.params.get(6).copied().unwrap_or(0.0),
                    self.params.get(7).copied().unwrap_or(0.0),
                ];
                project_equidistant(fx, fy, cx, cy, k, point_camera)
            }
            CameraModel::SimpleRadialFisheye => {
                let (fx, fy, cx, cy) = self.intrinsics()?;
                let k = [self.params.get(3).copied().unwrap_or(0.0), 0.0, 0.0, 0.0];
                project_equidistant(fx, fy, cx, cy, k, point_camera)
            }
            CameraModel::RadialFisheye => {
                let (fx, fy, cx, cy) = self.intrinsics()?;
                let k = [
                    self.params.get(3).copied().unwrap_or(0.0),
                    self.params.get(4).copied().unwrap_or(0.0),
                    0.0,
                    0.0,
                ];
                project_equidistant(fx, fy, cx, cy, k, point_camera)
            }
            CameraModel::Fov => {
                let [fx, fy, cx, cy] = params4(&self.params)?;
                let omega = *self.params.get(4)?;
                project_fov(fx, fy, cx, cy, omega, point_camera)
            }
            CameraModel::FullOpenCv => {
                let [fx, fy, cx, cy] = params4(&self.params)?;
                project_full_opencv(
                    fx,
                    fy,
                    cx,
                    cy,
                    full_opencv_coeffs(&self.params),
                    point_camera,
                )
            }
            CameraModel::OpenCv if self.tangential_distortion().is_some() => {
                let [fx, fy, cx, cy] = params4(&self.params)?;
                project_full_opencv(
                    fx,
                    fy,
                    cx,
                    cy,
                    full_opencv_coeffs(&self.params),
                    point_camera,
                )
            }
            _ => {
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
    }

    /// Back-project a pixel to a unit camera-frame ray.  Unlike
    /// [`Self::normalize_pixel`] this is defined for the full field of view,
    /// including rays at or behind the image plane, so it is the robust entry
    /// point for fisheye/wide-angle geometry.
    pub fn unit_ray_from_pixel(&self, point: &Point2<f64>) -> Option<Vector3<f64>> {
        if !point.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let (fx, fy, cx, cy) = self.intrinsics()?;
        match self.model {
            CameraModel::DoubleSphere => {
                let [_, _, _, _, xi, alpha] = params6(&self.params)?;
                unproject_double_sphere(fx, fy, cx, cy, xi, alpha, point)
            }
            CameraModel::OpenCvFisheye => {
                let k = [
                    self.params.get(4).copied().unwrap_or(0.0),
                    self.params.get(5).copied().unwrap_or(0.0),
                    self.params.get(6).copied().unwrap_or(0.0),
                    self.params.get(7).copied().unwrap_or(0.0),
                ];
                unproject_equidistant(fx, fy, cx, cy, k, point)
            }
            CameraModel::SimpleRadialFisheye => {
                let k = [self.params.get(3).copied().unwrap_or(0.0), 0.0, 0.0, 0.0];
                unproject_equidistant(fx, fy, cx, cy, k, point)
            }
            CameraModel::RadialFisheye => {
                let k = [
                    self.params.get(3).copied().unwrap_or(0.0),
                    self.params.get(4).copied().unwrap_or(0.0),
                    0.0,
                    0.0,
                ];
                unproject_equidistant(fx, fy, cx, cy, k, point)
            }
            CameraModel::Fov => {
                let omega = *self.params.get(4)?;
                unproject_fov(fx, fy, cx, cy, omega, point)
            }
            _ => {
                let ray = self.normalize_pixel(point)?;
                Vector3::new(ray.x, ray.y, 1.0).normalize().into()
            }
        }
    }
}

fn params4(params: &[f64]) -> Option<[f64; 4]> {
    Some([
        *params.first()?,
        *params.get(1)?,
        *params.get(2)?,
        *params.get(3)?,
    ])
}

fn params6(params: &[f64]) -> Option<[f64; 6]> {
    Some([
        *params.first()?,
        *params.get(1)?,
        *params.get(2)?,
        *params.get(3)?,
        *params.get(4)?,
        *params.get(5)?,
    ])
}

fn finite_point(point: Point2<f64>) -> Option<Point2<f64>> {
    point
        .coords
        .iter()
        .all(|value| value.is_finite())
        .then_some(point)
}

fn normalize_ray(ray: Vector3<f64>) -> Option<Vector3<f64>> {
    if !ray.iter().all(|value| value.is_finite()) {
        return None;
    }
    let norm = ray.norm();
    if !norm.is_finite() || norm <= EPS {
        return None;
    }
    Some(ray / norm)
}

/// Double Sphere projection (Basalt's `DoubleSphereCamera::project`).
fn project_double_sphere(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    xi: f64,
    alpha: f64,
    point: &Point3<f64>,
) -> Option<Point2<f64>> {
    let d1 = point.coords.norm();
    if d1 <= EPS {
        return None;
    }
    let zeta = xi * d1 + point.z;
    let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
    let denominator = alpha * d2 + (1.0 - alpha) * zeta;
    if !denominator.is_finite() || denominator <= EPS {
        return None;
    }
    finite_point(Point2::new(
        fx * point.x / denominator + cx,
        fy * point.y / denominator + cy,
    ))
}

/// Double Sphere unprojection to a unit ray (Basalt's
/// `DoubleSphereCamera::unproject`).
fn unproject_double_sphere(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    xi: f64,
    alpha: f64,
    point: &Point2<f64>,
) -> Option<Vector3<f64>> {
    let mx = (point.x - cx) / fx;
    let my = (point.y - cy) / fy;
    let r_sq = mx * mx + my * my;
    let sqrt_argument = 1.0 - (2.0 * alpha - 1.0) * r_sq;
    if !sqrt_argument.is_finite() || sqrt_argument < 0.0 {
        return None;
    }
    let mz_denominator = alpha * sqrt_argument.sqrt() + (1.0 - alpha);
    if mz_denominator.abs() <= EPS {
        return None;
    }
    let mz = (1.0 - alpha * alpha * r_sq) / mz_denominator;
    let k_radicand = mz * mz + (1.0 - xi * xi) * r_sq;
    if !k_radicand.is_finite() || k_radicand < 0.0 {
        return None;
    }
    let k_denominator = mz * mz + r_sq;
    if k_denominator <= EPS {
        return None;
    }
    let k = (mz * xi + k_radicand.sqrt()) / k_denominator;
    normalize_ray(Vector3::new(k * mx, k * my, k * mz - xi))
}

/// Kannala-Brandt equidistant `theta_d = theta (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ +
/// k4 θ⁸)` distortion factor.
fn equidistant_poly(theta: f64, k: [f64; 4]) -> f64 {
    let t2 = theta * theta;
    1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))
}

fn project_equidistant(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k: [f64; 4],
    point: &Point3<f64>,
) -> Option<Point2<f64>> {
    let r_xy = (point.x * point.x + point.y * point.y).sqrt();
    let theta = r_xy.atan2(point.z);
    let theta_d = theta * equidistant_poly(theta, k);
    let (x, y) = if r_xy <= EPS {
        (0.0, 0.0)
    } else {
        let scale = theta_d / r_xy;
        (scale * point.x, scale * point.y)
    };
    finite_point(Point2::new(fx * x + cx, fy * y + cy))
}

fn unproject_equidistant(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k: [f64; 4],
    point: &Point2<f64>,
) -> Option<Vector3<f64>> {
    let xd = (point.x - cx) / fx;
    let yd = (point.y - cy) / fy;
    let r_d = (xd * xd + yd * yd).sqrt();
    if r_d <= EPS {
        return normalize_ray(Vector3::new(0.0, 0.0, 1.0));
    }
    // Solve `theta * poly(theta) = r_d` by Newton from `theta = r_d`.
    let mut theta = r_d;
    for _ in 0..20 {
        let poly = equidistant_poly(theta, k);
        let residual = theta * poly - r_d;
        let t2 = theta * theta;
        let dpoly = 2.0 * theta * k[0]
            + 4.0 * t2 * theta * k[1]
            + 6.0 * t2 * t2 * theta * k[2]
            + 8.0 * t2 * t2 * t2 * theta * k[3];
        let derivative = poly + theta * dpoly;
        if !derivative.is_finite() || derivative.abs() <= EPS {
            break;
        }
        let next = theta - residual / derivative;
        if !next.is_finite() {
            break;
        }
        if (next - theta).abs() < 1e-14 {
            theta = next;
            break;
        }
        theta = next;
    }
    if !theta.is_finite() || !(0.0..std::f64::consts::PI).contains(&theta) {
        return None;
    }
    let radial = theta.tan();
    let scale = radial / r_d;
    normalize_ray(Vector3::new(scale * xd, scale * yd, 1.0))
}

/// Devernay FOV model: `r_d = atan(2 r tan(ω/2)) / ω`.
fn project_fov(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    omega: f64,
    point: &Point3<f64>,
) -> Option<Point2<f64>> {
    if omega.abs() <= EPS || point.z <= EPS {
        return None;
    }
    let xn = point.x / point.z;
    let yn = point.y / point.z;
    let r = (xn * xn + yn * yn).sqrt();
    let scale = if r <= EPS {
        1.0
    } else {
        (2.0 * r * (omega / 2.0).tan()).atan() / (omega * r)
    };
    finite_point(Point2::new(fx * scale * xn + cx, fy * scale * yn + cy))
}

fn unproject_fov(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    omega: f64,
    point: &Point2<f64>,
) -> Option<Vector3<f64>> {
    if omega.abs() <= EPS {
        return None;
    }
    let xd = (point.x - cx) / fx;
    let yd = (point.y - cy) / fy;
    let r_d = (xd * xd + yd * yd).sqrt();
    if r_d <= EPS {
        return normalize_ray(Vector3::new(0.0, 0.0, 1.0));
    }
    let r = (r_d * omega).tan() / (2.0 * (omega / 2.0).tan());
    if !r.is_finite() {
        return None;
    }
    let scale = r / r_d;
    normalize_ray(Vector3::new(scale * xd, scale * yd, 1.0))
}

/// `[k1, k2, p1, p2, k3, k4, k5, k6]` for the OpenCV rational model.
fn full_opencv_coeffs(params: &[f64]) -> [f64; 8] {
    let mut coeffs = [0.0_f64; 8];
    for (slot, value) in coeffs.iter_mut().zip(params.iter().skip(4)) {
        *slot = *value;
    }
    coeffs
}

fn project_full_opencv(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k: [f64; 8],
    point: &Point3<f64>,
) -> Option<Point2<f64>> {
    if point.z <= EPS {
        return None;
    }
    let x = point.x / point.z;
    let y = point.y / point.z;
    let r2 = x * x + y * y;
    let r2_2 = r2 * r2;
    let r2_3 = r2_2 * r2;
    let numerator = 1.0 + k[0] * r2 + k[1] * r2_2 + k[4] * r2_3;
    let denominator = 1.0 + k[5] * r2 + k[6] * r2_2 + k[7] * r2_3;
    if denominator.abs() <= EPS {
        return None;
    }
    let radial = numerator / denominator;
    let xd = x * radial + 2.0 * k[2] * x * y + k[3] * (r2 + 2.0 * x * x);
    let yd = y * radial + k[2] * (r2 + 2.0 * y * y) + 2.0 * k[3] * x * y;
    finite_point(Point2::new(fx * xd + cx, fy * yd + cy))
}

fn normalize_full_opencv(xd: f64, yd: f64, k: [f64; 8]) -> Point2<f64> {
    let (mut x, mut y) = (xd, yd);
    for _ in 0..30 {
        let r2 = x * x + y * y;
        let r2_2 = r2 * r2;
        let r2_3 = r2_2 * r2;
        let numerator = 1.0 + k[0] * r2 + k[1] * r2_2 + k[4] * r2_3;
        let denominator = 1.0 + k[5] * r2 + k[6] * r2_2 + k[7] * r2_3;
        if denominator.abs() <= EPS {
            break;
        }
        let radial = numerator / denominator;
        let dx = 2.0 * k[2] * x * y + k[3] * (r2 + 2.0 * x * x);
        let dy = k[2] * (r2 + 2.0 * y * y) + 2.0 * k[3] * x * y;
        let nx = (xd - dx) / radial;
        let ny = (yd - dy) / radial;
        if (nx - x).abs() + (ny - y).abs() < 1.0e-13 {
            return Point2::new(nx, ny);
        }
        x = nx;
        y = ny;
    }
    Point2::new(x, y)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(camera: &Camera, point: Point3<f64>) {
        let ray = point.coords.normalize();
        let pixel = camera
            .project(&point)
            .unwrap_or_else(|| panic!("project failed for {point:?}"));
        let recovered = camera
            .unit_ray_from_pixel(&pixel)
            .unwrap_or_else(|| panic!("unproject failed for {pixel:?}"));
        assert!(
            (recovered - ray).norm() < 1e-9,
            "camera {pixel:?}: ray={recovered:?} expected={ray:?}"
        );
        let pixel_again = camera.project(&recovered.into()).unwrap();
        assert!(
            (pixel_again - pixel).norm() < 1e-7,
            "camera {pixel:?}: reprojected={pixel_again:?}"
        );
    }

    #[test]
    fn opencv_fisheye_round_trips_across_field_of_view() {
        let camera = Camera::opencv_fisheye(
            0,
            640,
            480,
            300.0,
            300.0,
            320.0,
            240.0,
            [-0.02, 0.01, 0.0, 0.0],
        );
        for fov in [0.1_f64, 0.5, 1.0, 1.4] {
            round_trip(&camera, Point3::new(fov.sin(), 0.3 * fov.sin(), fov.cos()));
        }
    }

    #[test]
    fn simple_and_radial_fisheye_round_trip() {
        let simple = Camera {
            id: 1,
            model: CameraModel::SimpleRadialFisheye,
            width: 640,
            height: 480,
            params: vec![300.0, 320.0, 240.0, -0.05],
        };
        let radial = Camera {
            id: 2,
            model: CameraModel::RadialFisheye,
            width: 640,
            height: 480,
            params: vec![300.0, 320.0, 240.0, -0.05, 0.01],
        };
        for camera in [simple, radial] {
            let point = Point3::new(0.4, -0.2, 1.0);
            let ray = point.coords.normalize();
            let pixel = camera.project(&point).unwrap();
            let recovered = camera.unit_ray_from_pixel(&pixel).unwrap();
            assert!((recovered - ray).norm() < 1e-9, "{recovered:?} vs {ray:?}");
        }
    }

    #[test]
    fn fov_round_trips() {
        let camera = Camera {
            id: 3,
            model: CameraModel::Fov,
            width: 640,
            height: 480,
            params: vec![300.0, 300.0, 320.0, 240.0, 1.2],
        };
        round_trip(&camera, Point3::new(0.5, 0.2, 1.0));
        round_trip(&camera, Point3::new(-0.8, 0.1, 1.0));
    }

    #[test]
    fn double_sphere_round_trips_and_matches_basalt_golden() {
        let camera = Camera::double_sphere(
            4, 752, 480, 458.654, 457.296, 367.215, 248.375, 0.662073, 0.779792,
        );
        let pixel = camera.project(&Point3::new(0.1, -0.05, 1.0)).unwrap();
        assert!((pixel.x - 394.693_793_482_414_3).abs() < 1e-9, "{pixel:?}");
        assert!((pixel.y - 234.676_283_381_008_18).abs() < 1e-9, "{pixel:?}");
        round_trip(&camera, Point3::new(0.1, -0.05, 1.0));
        // Rays behind the image plane are representable as unit rays even
        // though `normalize_pixel` cannot express them.
        let wide = camera.project(&Point3::new(1.5, -0.5, 0.2)).unwrap();
        assert!(camera.unit_ray_from_pixel(&wide).is_some());
    }

    #[test]
    fn full_opencv_round_trips_and_names() {
        let camera = Camera {
            id: 5,
            model: CameraModel::FullOpenCv,
            width: 640,
            height: 480,
            params: vec![
                500.0, 500.0, 320.0, 240.0, 0.05, -0.02, 0.001, -0.001, 0.01, 0.0, 0.0, 0.0,
            ],
        };
        for point in [Point3::new(0.2, -0.1, 1.0), Point3::new(-0.4, 0.3, 1.5)] {
            round_trip(&camera, point);
        }
        assert_eq!(camera.model.colmap_name(), Some("FULL_OPENCV"));
        assert!(!camera.model.is_fisheye());
    }

    #[test]
    fn colmap_names_round_trip_for_fisheye() {
        for name in [
            "OPENCV_FISHEYE",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "FOV",
        ] {
            let model = CameraModel::from_colmap_name(name);
            assert_eq!(model.colmap_name(), Some(name));
            assert!(model.is_fisheye());
        }
        assert_eq!(CameraModel::DoubleSphere.colmap_name(), None);
    }

    fn opencv(p1: f64, p2: f64) -> Camera {
        Camera {
            id: 0,
            model: CameraModel::OpenCv,
            width: 1920,
            height: 1080,
            params: vec![1353.09, 1338.03, 962.7, 539.93, 0.0346, -0.0235, p1, p2],
        }
    }

    #[test]
    fn opencv_tangential_matches_cv2_project_points() {
        // Reference values from OpenCV 4.10 `cv2.projectPoints` with
        // K = [[1353.09, 0, 962.7], [0, 1338.03, 539.93]] and
        // dist = [k1, k2, p1, p2] = [0.0346, -0.0235, 0.0012, -0.0008].
        let camera = opencv(0.0012, -0.0008);
        assert_eq!(camera.tangential_distortion(), Some((0.0012, -0.0008)));
        for (point, (u, v)) in [
            (
                Point3::new(0.5, -0.3, 1.0),
                (1643.9695843529998, 136.0453501334),
            ),
            (
                Point3::new(-0.7, 0.4, 1.2),
                (162.9927043515022, 992.2680873586517),
            ),
            (
                Point3::new(0.6, 0.35, 0.9),
                (1875.0649158305705, 1067.547689735487),
            ),
        ] {
            let pixel = camera.project(&point).unwrap();
            assert!(
                (pixel.x - u).abs() < 1e-8 && (pixel.y - v).abs() < 1e-8,
                "{point:?}: got {pixel:?}, cv2 ({u}, {v})"
            );
            round_trip(&camera, point);
        }
    }

    #[test]
    fn opencv_without_tangential_keeps_the_radial_only_path_bit_exact() {
        // p1 = p2 = 0 must not change results for existing OPENCV / radial users.
        let opencv = opencv(0.0, 0.0);
        let radial = Camera::pinhole_radial(
            0, 1920, 1080, 1353.09, 1338.03, 962.7, 539.93, 0.0346, -0.0235,
        );
        assert_eq!(opencv.tangential_distortion(), None);
        for point in [Point3::new(0.5, -0.3, 1.0), Point3::new(-0.7, 0.4, 1.2)] {
            let a = opencv.project(&point).unwrap();
            assert_eq!(a, radial.project(&point).unwrap());
            assert_eq!(opencv.normalize_pixel(&a), radial.normalize_pixel(&a));
        }
    }
}
