//! A 2D occupancy grid with the conventions of ROS `nav_msgs/OccupancyGrid`
//! and reading / writing in the `map_server` format that Nav2 loads.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Cell value of an unknown cell, as in `nav_msgs/OccupancyGrid`.
pub const UNKNOWN: i8 = -1;
/// Cell value of a free cell.
pub const FREE: i8 = 0;
/// Cell value of an occupied cell.
pub const OCCUPIED: i8 = 100;

/// Where a grid lies in the map frame.
///
/// Cell `(ix, iy)` covers `[x0 + ix * r, x0 + (ix + 1) * r) x
/// [y0 + iy * r, y0 + (iy + 1) * r)` for origin `(x0, y0)` and resolution
/// `r`; row `iy = 0` is the lowest `y`, as in `nav_msgs/OccupancyGrid`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridGeometry {
    /// Cell edge length in metres.
    pub resolution: f64,
    /// Map-frame `x` of the lower-left corner of cell `(0, 0)`.
    pub origin_x: f64,
    /// Map-frame `y` of the lower-left corner of cell `(0, 0)`.
    pub origin_y: f64,
    pub width: usize,
    pub height: usize,
}

impl GridGeometry {
    /// Number of cells.
    pub const fn len(&self) -> usize {
        self.width * self.height
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major index of a cell (row `iy`, column `ix`).
    pub const fn index(&self, ix: usize, iy: usize) -> usize {
        iy * self.width + ix
    }

    /// The cell holding a map-frame point, if it is inside the grid.
    pub fn cell_of(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let fx = ((x - self.origin_x) / self.resolution).floor();
        let fy = ((y - self.origin_y) / self.resolution).floor();
        if !(fx.is_finite() && fy.is_finite()) || fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (ix, iy) = (fx as usize, fy as usize);
        (ix < self.width && iy < self.height).then_some((ix, iy))
    }

    /// Map-frame centre of a cell.
    pub fn cell_center(&self, ix: usize, iy: usize) -> (f64, f64) {
        (
            self.origin_x + (ix as f64 + 0.5) * self.resolution,
            self.origin_y + (iy as f64 + 0.5) * self.resolution,
        )
    }
}

/// A 2D occupancy grid: [`UNKNOWN`], [`FREE`] or [`OCCUPIED`] (or any value
/// in `0..=100` read from elsewhere) per cell, row-major from the lowest `y`.
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyGrid {
    pub geometry: GridGeometry,
    pub data: Vec<i8>,
}

/// Cell counts of a grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GridCounts {
    pub free: usize,
    pub occupied: usize,
    pub unknown: usize,
}

impl OccupancyGrid {
    /// A grid with every cell unknown.
    pub fn unknown(geometry: GridGeometry) -> Self {
        Self {
            geometry,
            data: vec![UNKNOWN; geometry.len()],
        }
    }

    pub fn get(&self, ix: usize, iy: usize) -> i8 {
        self.data[self.geometry.index(ix, iy)]
    }

    pub fn set(&mut self, ix: usize, iy: usize, value: i8) {
        let index = self.geometry.index(ix, iy);
        self.data[index] = value;
    }

    /// The value at a map-frame point; `None` outside the grid.
    pub fn value_at(&self, x: f64, y: f64) -> Option<i8> {
        self.geometry.cell_of(x, y).map(|(ix, iy)| self.get(ix, iy))
    }

    pub fn counts(&self) -> GridCounts {
        let mut counts = GridCounts::default();
        for &value in &self.data {
            match value {
                v if v < 0 => counts.unknown += 1,
                v if v >= 50 => counts.occupied += 1,
                _ => counts.free += 1,
            }
        }
        counts
    }

    /// The smallest sub-grid holding every known cell, grown by `margin`
    /// cells on each side (clamped to the grid). A grid with no known cell
    /// is returned unchanged.
    pub fn cropped_to_known(&self, margin: usize) -> Self {
        let g = self.geometry;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for iy in 0..g.height {
            for ix in 0..g.width {
                if self.get(ix, iy) >= 0 {
                    x0 = x0.min(ix);
                    y0 = y0.min(iy);
                    x1 = x1.max(ix);
                    y1 = y1.max(iy);
                }
            }
        }
        if x0 == usize::MAX {
            return self.clone();
        }
        let x0 = x0.saturating_sub(margin);
        let y0 = y0.saturating_sub(margin);
        let x1 = (x1 + margin).min(g.width - 1);
        let y1 = (y1 + margin).min(g.height - 1);
        let geometry = GridGeometry {
            resolution: g.resolution,
            origin_x: g.origin_x + x0 as f64 * g.resolution,
            origin_y: g.origin_y + y0 as f64 * g.resolution,
            width: x1 - x0 + 1,
            height: y1 - y0 + 1,
        };
        let mut cropped = Self::unknown(geometry);
        for iy in 0..geometry.height {
            for ix in 0..geometry.width {
                cropped.set(ix, iy, self.get(ix + x0, iy + y0));
            }
        }
        cropped
    }

    /// Writes `<stem>.pgm` and `<stem>.yaml` in the ROS `map_server` format
    /// (trinary mode: occupied black, free white, unknown grey), which Nav2's
    /// `map_server` loads with `yaml_filename:=<stem>.yaml`. Returns the
    /// YAML path.
    pub fn write_map_server(&self, stem: impl AsRef<Path>) -> io::Result<PathBuf> {
        let stem = stem.as_ref();
        let pgm_path = stem.with_extension("pgm");
        let yaml_path = stem.with_extension("yaml");
        if let Some(parent) = pgm_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(&pgm_path, self.to_pgm())?;
        let image_name = pgm_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        fs::write(&yaml_path, self.map_server_yaml(&image_name))?;
        Ok(yaml_path)
    }

    /// The binary PGM (`P5`) image of `map_server`'s trinary mode: the first
    /// image row is the grid's highest `y`.
    pub fn to_pgm(&self) -> Vec<u8> {
        let g = self.geometry;
        let mut bytes = format!(
            "P5\n# visloc-navigation occupancy grid\n{} {}\n255\n",
            g.width, g.height
        )
        .into_bytes();
        bytes.reserve(g.len());
        for row in (0..g.height).rev() {
            for col in 0..g.width {
                bytes.push(match self.get(col, row) {
                    v if v < 0 => 205,
                    v if v >= 65 => 0,
                    v if v <= 25 => 254,
                    _ => 205,
                });
            }
        }
        bytes
    }

    /// The `map_server` YAML for an image named `image_name` next to it.
    pub fn map_server_yaml(&self, image_name: &str) -> String {
        let g = self.geometry;
        format!(
            "image: {image_name}\nmode: trinary\nresolution: {}\norigin: [{}, {}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
            g.resolution, g.origin_x, g.origin_y
        )
    }

    /// Reads a `map_server` map (YAML plus the PGM it names) in trinary
    /// semantics: a pixel darker than `occupied_thresh` is occupied, one
    /// lighter than `free_thresh` free, anything else unknown. Only binary
    /// PGM images and an origin yaw of zero are supported.
    pub fn read_map_server(yaml_path: impl AsRef<Path>) -> Result<Self, MapServerError> {
        let yaml_path = yaml_path.as_ref();
        let yaml = fs::read_to_string(yaml_path)?;
        let mut image = None;
        let mut resolution = None;
        let mut origin = None;
        let mut negate = false;
        let mut occupied_thresh = 0.65;
        let mut free_thresh = 0.196;
        for line in yaml.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
            let number = |v: &str| {
                v.parse::<f64>()
                    .map_err(|_| MapServerError::Format(format!("bad {key}: {v}")))
            };
            match key.trim() {
                "image" => image = Some(value.to_owned()),
                "resolution" => resolution = Some(number(value)?),
                "negate" => negate = value == "1" || value.eq_ignore_ascii_case("true"),
                "occupied_thresh" => occupied_thresh = number(value)?,
                "free_thresh" => free_thresh = number(value)?,
                "origin" => {
                    let values = value
                        .trim_start_matches('[')
                        .trim_end_matches(']')
                        .split(',')
                        .map(|v| number(v.trim()))
                        .collect::<Result<Vec<_>, _>>()?;
                    if values.len() != 3 {
                        return Err(MapServerError::Format("origin needs 3 values".into()));
                    }
                    if values[2].abs() > 1e-9 {
                        return Err(MapServerError::Format(
                            "origin yaw other than 0 is not supported".into(),
                        ));
                    }
                    origin = Some((values[0], values[1]));
                }
                _ => {}
            }
        }
        let missing = |what: &str| MapServerError::Format(format!("missing {what}"));
        let image = image.ok_or_else(|| missing("image"))?;
        let resolution = resolution.ok_or_else(|| missing("resolution"))?;
        let (origin_x, origin_y) = origin.ok_or_else(|| missing("origin"))?;
        let image_path = yaml_path
            .parent()
            .map_or_else(|| PathBuf::from(&image), |dir| dir.join(&image));
        let (width, height, pixels) = parse_pgm(&fs::read(&image_path)?)?;
        let geometry = GridGeometry {
            resolution,
            origin_x,
            origin_y,
            width,
            height,
        };
        let mut grid = Self::unknown(geometry);
        for row in 0..height {
            for col in 0..width {
                let pixel = f64::from(pixels[row * width + col]) / 255.0;
                let occupancy = if negate { pixel } else { 1.0 - pixel };
                let value = if occupancy > occupied_thresh {
                    OCCUPIED
                } else if occupancy < free_thresh {
                    FREE
                } else {
                    UNKNOWN
                };
                grid.set(col, height - 1 - row, value);
            }
        }
        Ok(grid)
    }
}

/// Error reading a `map_server` map.
#[derive(Debug)]
pub enum MapServerError {
    Io(io::Error),
    Format(String),
}

impl fmt::Display for MapServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Format(message) => write!(f, "map_server map: {message}"),
        }
    }
}

impl std::error::Error for MapServerError {}

impl From<io::Error> for MapServerError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Parses a binary 8-bit PGM (`P5`): width, height, row-major pixels.
fn parse_pgm(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), MapServerError> {
    let bad = |what: &str| MapServerError::Format(format!("PGM: {what}"));
    let mut fields = Vec::new();
    let mut pos = 0;
    while fields.len() < 4 {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if start == pos {
            return Err(bad("truncated header"));
        }
        fields.push(String::from_utf8_lossy(&bytes[start..pos]).into_owned());
    }
    if fields[0] != "P5" {
        return Err(bad("only binary P5 images are supported"));
    }
    let width: usize = fields[1].parse().map_err(|_| bad("width"))?;
    let height: usize = fields[2].parse().map_err(|_| bad("height"))?;
    if fields[3] != "255" {
        return Err(bad("only 8-bit images are supported"));
    }
    // Exactly one whitespace byte separates the header from the pixels.
    let data = bytes.get(pos + 1..).ok_or_else(|| bad("no pixels"))?;
    if data.len() < width * height {
        return Err(bad("fewer pixels than width x height"));
    }
    Ok((width, height, data[..width * height].to_vec()))
}
