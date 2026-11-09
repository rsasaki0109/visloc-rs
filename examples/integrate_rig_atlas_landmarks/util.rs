//! Line parsing and small formatting helpers.

use super::*;

pub(super) fn bounded_sample(keys: &[ObservationKey], max_count: usize) -> Vec<ObservationKey> {
    if keys.len() <= max_count {
        return keys.to_vec();
    }
    let mut result = Vec::with_capacity(max_count);
    for index in 0..max_count {
        let source_index = index * (keys.len() - 1) / (max_count - 1);
        if result.last().copied() != Some(keys[source_index]) {
            result.push(keys[source_index]);
        }
    }
    result
}

pub(super) fn points_close(left: &Point2<f64>, right: &Point2<f64>) -> bool {
    (left - right).norm() <= XY_TOLERANCE_PX
        && left.coords.iter().all(|value| value.is_finite())
        && right.coords.iter().all(|value| value.is_finite())
}

pub(super) fn parse_points2d(
    line: &str,
    path: &Path,
    line_number: usize,
) -> Result<Vec<SourceKeypoint>, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() % 3 != 0 {
        return Err(format!(
            "COLMAP {}:{} POINTS2D is not triples",
            path.display(),
            line_number
        ));
    }
    let mut points = Vec::with_capacity(fields.len() / 3);
    for triple in fields.chunks_exact(3) {
        let x = parse_f64(triple[0], "POINTS2D x", path, line_number)?;
        let y = parse_f64(triple[1], "POINTS2D y", path, line_number)?;
        let point_id = triple[2].parse::<i64>().map_err(|error| {
            format!(
                "COLMAP {}:{} invalid POINT3D_ID: {error}",
                path.display(),
                line_number
            )
        })?;
        if point_id < -1 {
            return Err(format!(
                "COLMAP {}:{} POINT3D_ID is below -1",
                path.display(),
                line_number
            ));
        }
        points.push(SourceKeypoint {
            xy: Point2::new(x, y),
            point3d_id: (point_id >= 0).then_some(point_id as u64),
        });
    }
    Ok(points)
}

pub(super) fn next_data_line<'a>(
    lines: &'a [&'a str],
    index: &mut usize,
) -> Option<(usize, &'a str)> {
    while *index < lines.len() {
        let line_number = *index + 1;
        let line = lines[*index].trim();
        *index += 1;
        if !line.is_empty() && !line.starts_with('#') {
            return Some((line_number, line));
        }
    }
    None
}

pub(super) fn next_required_line<'a>(
    lines: &'a [&'a str],
    index: &mut usize,
) -> Option<(usize, &'a str)> {
    if *index >= lines.len() {
        return None;
    }
    let line_number = *index + 1;
    let line = lines[*index].trim();
    *index += 1;
    if line.starts_with('#') {
        None
    } else {
        Some((line_number, line))
    }
}

pub(super) fn parse_quaternion(
    fields: &[&str],
    path: &Path,
    line: usize,
) -> Result<UnitQuaternion<f64>, String> {
    if fields.len() != 4 {
        return Err(format!(
            "COLMAP {}:{} quaternion requires four values",
            path.display(),
            line
        ));
    }
    let values = fields
        .iter()
        .map(|field| parse_f64(field, "quaternion", path, line))
        .collect::<Result<Vec<_>, _>>()?;
    let quaternion = Quaternion::new(values[0], values[1], values[2], values[3]);
    let norm = quaternion.norm();
    if !norm.is_finite() || norm <= MIN_HOMOGENEOUS_SCALE {
        return Err(format!(
            "COLMAP {}:{} quaternion is invalid",
            path.display(),
            line
        ));
    }
    Ok(UnitQuaternion::new_normalize(quaternion))
}

pub(super) fn parse_vector(
    fields: &[&str],
    path: &Path,
    line: usize,
) -> Result<Vector3<f64>, String> {
    if fields.len() != 3 {
        return Err(format!(
            "COLMAP {}:{} translation requires three values",
            path.display(),
            line
        ));
    }
    Ok(Vector3::new(
        parse_f64(fields[0], "translation", path, line)?,
        parse_f64(fields[1], "translation", path, line)?,
        parse_f64(fields[2], "translation", path, line)?,
    ))
}

pub(super) fn parse_u64(value: &str, label: &str, path: &Path, line: usize) -> Result<u64, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

pub(super) fn parse_u32(value: &str, label: &str, path: &Path, line: usize) -> Result<u32, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

pub(super) fn parse_usize(
    value: &str,
    label: &str,
    path: &Path,
    line: usize,
) -> Result<usize, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

pub(super) fn parse_f64(value: &str, label: &str, path: &Path, line: usize) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })?;
    if !parsed.is_finite() {
        return Err(format!("{}:{} non-finite {label}", path.display(), line));
    }
    Ok(parsed)
}

pub(super) fn resolve_path(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

pub(super) fn camera_model_name(model: &CameraModel) -> Result<&'static str, String> {
    match model {
        CameraModel::Pinhole => Ok("PINHOLE"),
        CameraModel::SimplePinhole => Ok("SIMPLE_PINHOLE"),
        CameraModel::SimpleRadial => Ok("SIMPLE_RADIAL"),
        CameraModel::Radial => Ok("RADIAL"),
        CameraModel::OpenCv => Ok("OPENCV"),
        CameraModel::FullOpenCv => Ok("FULL_OPENCV"),
        CameraModel::OpenCvFisheye => Ok("OPENCV_FISHEYE"),
        CameraModel::SimpleRadialFisheye => Ok("SIMPLE_RADIAL_FISHEYE"),
        CameraModel::RadialFisheye => Ok("RADIAL_FISHEYE"),
        CameraModel::ThinPrismFisheye => Ok("THIN_PRISM_FISHEYE"),
        CameraModel::Fov => Ok("FOV"),
        CameraModel::Equirectangular => Ok("EQUIRECTANGULAR"),
        CameraModel::DoubleSphere => Err("Double Sphere is not a COLMAP model".to_owned()),
        CameraModel::Unknown(name) => Err(format!("unsupported camera model {name:?}")),
    }
}

pub(super) fn validate_image_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte == 0)
    {
        return Err(format!(
            "image name is not representable in COLMAP text: {name:?}"
        ));
    }
    Ok(())
}

pub(super) fn format_f64(value: f64) -> String {
    let formatted = format!("{value:.12}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

pub(super) fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted.get(index.min(sorted.len() - 1)).copied()
}
