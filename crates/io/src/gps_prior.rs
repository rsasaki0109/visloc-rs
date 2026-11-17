//! GPS position priors for SfM: EXIF GPS tags, a plain-text prior file, and
//! their conversion to a local East-North-Up (ENU) frame.
//!
//! The SfM side (`visloc_slam::incremental_sfm::SfmPositionPriors`) wants a
//! metric Cartesian camera centre per image. GPS gives WGS84 latitude,
//! longitude and altitude; [`GpsPriorSet::to_local_enu`] turns those into
//! metres in the ENU tangent frame at an origin (by default the mean fix),
//! with the exact WGS84 → ECEF → ENU chain of [`crate::kitti_raw_gnss`].
//!
//! Prior file (`#` comments, whitespace- or comma-separated):
//!
//! ```text
//! # NAME            LAT          LON           ALT     [STD_H [STD_V]]
//! IMG_0001.JPG      35.681236    139.767125    40.2    3.0  6.0
//! ```
//!
//! With [`PriorCoordinates::Cartesian`] the three numbers are `X Y Z` in an
//! already metric frame (COLMAP `model_aligner`'s `ref_images` layout) and
//! are used as they are. `NAME` is matched against the image file name, its
//! path relative to the image root, or its stem.

use std::collections::BTreeMap;
use std::path::Path;

use nalgebra::{Point3, Vector3};

use crate::kitti_raw_gnss::LocalEnuFrame;

/// A WGS84 fix: degrees, degrees, metres above the ellipsoid (EXIF GPS stores
/// metres above sea level; the difference, the geoid height, is a constant
/// offset over a capture and does not matter to the SfM).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodeticPosition {
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// `None` when the source carries no altitude.
    pub alt_m: Option<f64>,
}

/// How the three coordinates of a prior are to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorCoordinates {
    /// WGS84 latitude, longitude (degrees), altitude (metres).
    Wgs84,
    /// Already metric Cartesian `X Y Z`.
    Cartesian,
}

impl std::str::FromStr for PriorCoordinates {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "wgs84" | "gps" | "lla" => Ok(Self::Wgs84),
            "cartesian" | "xyz" | "enu" => Ok(Self::Cartesian),
            other => Err(format!(
                "unknown prior coordinates {other:?} (expected wgs84|cartesian)"
            )),
        }
    }
}

/// One image's raw prior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsPrior {
    /// `[lat, lon, alt]` for [`PriorCoordinates::Wgs84`], `[x, y, z]` for
    /// [`PriorCoordinates::Cartesian`]. A missing altitude is `NaN`.
    pub coords: [f64; 3],
    /// Horizontal 1-σ in metres, when the source states it.
    pub std_horizontal_m: Option<f64>,
    /// Vertical 1-σ in metres, when the source states it.
    pub std_vertical_m: Option<f64>,
}

/// A metric prior in the local frame, ready for the SfM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalPositionPrior {
    pub position: Point3<f64>,
    /// Per-axis 1-σ (east, north, up for ENU). An infinite entry means the
    /// axis is unknown (e.g. no altitude) and carries no weight.
    pub std_dev: Vector3<f64>,
}

/// Priors keyed by image name.
#[derive(Debug, Clone, PartialEq)]
pub struct GpsPriorSet {
    pub coordinates: PriorCoordinates,
    pub priors: BTreeMap<String, GpsPrior>,
}

/// The local frame priors were converted into.
#[derive(Debug, Clone, PartialEq)]
pub enum PriorFrame {
    /// ENU at this geodetic origin.
    Enu(LocalEnuFrame),
    /// The file's own Cartesian frame.
    Cartesian,
}

impl PriorFrame {
    /// A short text description, e.g. for an `enu_origin.txt` next to a model.
    pub fn describe(&self) -> String {
        match self {
            Self::Enu(frame) => format!(
                "# Local ENU frame (x east, y north, z up, metres) of the model.\n\
                 # ORIGIN_LAT_DEG ORIGIN_LON_DEG ORIGIN_ALT_M\n{:.9} {:.9} {:.4}\n",
                frame.origin_lat_deg, frame.origin_lon_deg, frame.origin_alt_m
            ),
            Self::Cartesian => "# The model is in the prior file's Cartesian frame.\n".to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpsPriorError {
    #[error("read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}:{line}: {message}")]
    Parse {
        path: String,
        line: usize,
        message: String,
    },
}

impl GpsPriorSet {
    pub const fn new(coordinates: PriorCoordinates) -> Self {
        Self {
            coordinates,
            priors: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.priors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.priors.is_empty()
    }

    /// Parse a prior file's text (see the module docs). `path` is only used
    /// in error messages.
    pub fn parse(
        text: &str,
        coordinates: PriorCoordinates,
        path: &str,
    ) -> Result<Self, GpsPriorError> {
        let mut set = Self::new(coordinates);
        for (index, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let err = |message: String| GpsPriorError::Parse {
                path: path.to_string(),
                line: index + 1,
                message,
            };
            let fields: Vec<&str> = line
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|f| !f.is_empty())
                .collect();
            if fields.len() < 4 {
                return Err(err(format!(
                    "expected NAME C1 C2 C3 [STD_H [STD_V]], got {} fields",
                    fields.len()
                )));
            }
            let mut numbers = Vec::with_capacity(fields.len() - 1);
            for field in &fields[1..] {
                numbers.push(
                    field
                        .parse::<f64>()
                        .map_err(|e| err(format!("{field:?}: {e}")))?,
                );
            }
            if numbers.len() > 5 {
                return Err(err(format!(
                    "{} numbers, at most 5 expected",
                    numbers.len()
                )));
            }
            if coordinates == PriorCoordinates::Wgs84
                && (numbers[0].abs() > 90.0 || numbers[1].abs() > 180.0)
            {
                return Err(err(format!(
                    "latitude/longitude out of range: {} {}",
                    numbers[0], numbers[1]
                )));
            }
            let std_horizontal_m = numbers.get(3).copied();
            let std_vertical_m = numbers.get(4).copied().or(std_horizontal_m);
            set.priors.insert(
                fields[0].to_string(),
                GpsPrior {
                    coords: [numbers[0], numbers[1], numbers[2]],
                    std_horizontal_m,
                    std_vertical_m,
                },
            );
        }
        Ok(set)
    }

    /// Read and parse a prior file.
    pub fn read(path: &Path, coordinates: PriorCoordinates) -> Result<Self, GpsPriorError> {
        let text = std::fs::read_to_string(path).map_err(|source| GpsPriorError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&text, coordinates, &path.display().to_string())
    }

    /// Insert a geodetic fix (e.g. from [`exif_gps`]). A missing altitude is
    /// stored as `NaN` and later leaves the vertical axis unweighted.
    pub fn insert_geodetic(&mut self, name: impl Into<String>, fix: GeodeticPosition) {
        self.priors.insert(
            name.into(),
            GpsPrior {
                coords: [fix.lat_deg, fix.lon_deg, fix.alt_m.unwrap_or(f64::NAN)],
                std_horizontal_m: None,
                std_vertical_m: None,
            },
        );
    }

    /// Find the prior of an image by its name as given, its file name, or
    /// its stem.
    pub fn lookup(&self, image_name: &str) -> Option<&GpsPrior> {
        if let Some(prior) = self.priors.get(image_name) {
            return Some(prior);
        }
        let path = Path::new(image_name);
        let file_name = path.file_name().and_then(|s| s.to_str());
        let stem = path.file_stem().and_then(|s| s.to_str());
        file_name
            .and_then(|f| self.priors.get(f))
            .or_else(|| stem.and_then(|s| self.priors.get(s)))
            .or_else(|| {
                // Prior keyed by a path or a name with an extension, image by stem.
                self.priors.iter().find_map(|(key, prior)| {
                    let key_path = Path::new(key);
                    let matches = key_path.file_name().and_then(|s| s.to_str()) == file_name
                        || key_path.file_stem().and_then(|s| s.to_str()) == stem;
                    matches.then_some(prior)
                })
            })
    }

    /// The ENU origin used by [`Self::to_local_enu`] when none is given: the
    /// mean latitude/longitude/altitude of the priors.
    pub fn mean_origin(&self) -> Option<GeodeticPosition> {
        if self.coordinates != PriorCoordinates::Wgs84 || self.priors.is_empty() {
            return None;
        }
        let n = self.priors.len() as f64;
        let lat = self.priors.values().map(|p| p.coords[0]).sum::<f64>() / n;
        let lon = self.priors.values().map(|p| p.coords[1]).sum::<f64>() / n;
        let alts: Vec<f64> = self
            .priors
            .values()
            .map(|p| p.coords[2])
            .filter(|a| a.is_finite())
            .collect();
        let alt = (!alts.is_empty()).then(|| alts.iter().sum::<f64>() / alts.len() as f64);
        Some(GeodeticPosition {
            lat_deg: lat,
            lon_deg: lon,
            alt_m: alt,
        })
    }

    /// Metric priors for `image_names` (indexed like them) in a local frame:
    /// ENU at `origin` (default: [`Self::mean_origin`]) for WGS84 priors, the
    /// file's frame for Cartesian ones. A prior without a stated σ gets
    /// `default_std_m` (east, north, up); one without an altitude gets an
    /// unweighted vertical axis.
    pub fn to_local_enu(
        &self,
        image_names: &[String],
        origin: Option<GeodeticPosition>,
        default_std_m: Vector3<f64>,
    ) -> (Vec<Option<LocalPositionPrior>>, PriorFrame) {
        let frame = match self.coordinates {
            PriorCoordinates::Cartesian => PriorFrame::Cartesian,
            PriorCoordinates::Wgs84 => {
                let origin = origin
                    .or_else(|| self.mean_origin())
                    .unwrap_or(GeodeticPosition {
                        lat_deg: 0.0,
                        lon_deg: 0.0,
                        alt_m: None,
                    });
                PriorFrame::Enu(LocalEnuFrame::new(
                    origin.lat_deg,
                    origin.lon_deg,
                    origin.alt_m.unwrap_or(0.0),
                ))
            }
        };
        let priors = image_names
            .iter()
            .map(|name| {
                let prior = self.lookup(name)?;
                let (std_e, std_n) = prior
                    .std_horizontal_m
                    .map_or((default_std_m.x, default_std_m.y), |s| (s, s));
                let std_v = prior.std_vertical_m.unwrap_or(default_std_m.z);
                let has_alt = prior.coords[2].is_finite();
                let position = match &frame {
                    PriorFrame::Cartesian => Point3::new(
                        prior.coords[0],
                        prior.coords[1],
                        if has_alt { prior.coords[2] } else { 0.0 },
                    ),
                    PriorFrame::Enu(enu) => {
                        let alt = if has_alt {
                            prior.coords[2]
                        } else {
                            enu.origin_alt_m
                        };
                        Point3::from(enu.geodetic_to_enu(prior.coords[0], prior.coords[1], alt))
                    }
                };
                Some(LocalPositionPrior {
                    position,
                    std_dev: Vector3::new(
                        std_e,
                        std_n,
                        if has_alt { std_v } else { f64::INFINITY },
                    ),
                })
            })
            .collect();
        (priors, frame)
    }
}

/// The GPS fix in a JPEG's EXIF (`GPSLatitude`/`GPSLongitude` with their
/// references, and `GPSAltitude`/`GPSAltitudeRef` when present), or `None`
/// when the file has no EXIF GPS block or it is malformed.
pub fn exif_gps(bytes: &[u8]) -> Option<GeodeticPosition> {
    let tiff = jpeg_exif_tiff(bytes)?;
    let le = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = [*tiff.get(o)?, *tiff.get(o + 1)?];
        Some(if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b: [u8; 4] = tiff.get(o..o + 4)?.try_into().ok()?;
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    // IFD entry of `tag`: (type, count, offset of the value/offset field).
    let entry = |ifd: usize, tag: u16| -> Option<(u16, u32, usize)> {
        let n = u16_at(ifd)? as usize;
        (0..n)
            .map(|k| ifd + 2 + 12 * k)
            .find_map(|e| (u16_at(e)? == tag).then_some((u16_at(e + 2)?, u32_at(e + 4)?, e + 8)))
    };
    // Values over four bytes live at the offset the entry holds.
    let rationals = |ifd: usize, tag: u16| -> Option<Vec<f64>> {
        let (kind, count, field) = entry(ifd, tag)?;
        if kind != 5 || count == 0 || count > 8 {
            return None;
        }
        let base = u32_at(field)? as usize;
        (0..count as usize)
            .map(|k| {
                let num = u32_at(base + 8 * k)? as f64;
                let den = u32_at(base + 8 * k + 4)? as f64;
                (den != 0.0).then_some(num / den)
            })
            .collect()
    };
    let ascii_byte = |ifd: usize, tag: u16| -> Option<u8> {
        let (_, _, field) = entry(ifd, tag)?;
        tiff.get(field).copied()
    };
    let ifd0 = u32_at(4)? as usize;
    let (_, _, gps_field) = entry(ifd0, 0x8825)?;
    let gps = u32_at(gps_field)? as usize;
    let dms = |values: Vec<f64>| -> f64 {
        values.first().copied().unwrap_or(0.0)
            + values.get(1).copied().unwrap_or(0.0) / 60.0
            + values.get(2).copied().unwrap_or(0.0) / 3600.0
    };
    let mut lat = dms(rationals(gps, 0x0002)?);
    let mut lon = dms(rationals(gps, 0x0004)?);
    if ascii_byte(gps, 0x0001) == Some(b'S') {
        lat = -lat;
    }
    if ascii_byte(gps, 0x0003) == Some(b'W') {
        lon = -lon;
    }
    if !(lat.is_finite() && lon.is_finite() && lat.abs() <= 90.0 && lon.abs() <= 180.0) {
        return None;
    }
    // Cameras without a fix often write 0/0; treat that as no fix.
    if lat == 0.0 && lon == 0.0 {
        return None;
    }
    let alt_m = rationals(gps, 0x0006)
        .and_then(|v| v.first().copied())
        .filter(|a| a.is_finite())
        .map(|a| match entry(gps, 0x0005) {
            Some((_, _, field)) if tiff.get(field) == Some(&1) => -a,
            _ => a,
        });
    Some(GeodeticPosition {
        lat_deg: lat,
        lon_deg: lon,
        alt_m,
    })
}

/// The TIFF block of a JPEG's first `APP1` `Exif` segment.
fn jpeg_exif_tiff(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.get(0..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut p = 2usize;
    loop {
        if *bytes.get(p)? != 0xFF {
            return None;
        }
        let marker = *bytes.get(p + 1)?;
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            p += 2;
            continue;
        }
        let len = u16::from_be_bytes([*bytes.get(p + 2)?, *bytes.get(p + 3)?]) as usize;
        if len < 2 {
            return None;
        }
        if marker == 0xE1 && bytes.get(p + 4..p + 10)? == b"Exif\0\0" {
            return bytes.get(p + 10..p + 2 + len);
        }
        if marker == 0xDA || marker == 0xD9 {
            return None;
        }
        p += 2 + len;
    }
}

/// [`exif_gps`] of the file at `path`; `None` when it cannot be read or has
/// no GPS. Reads only the head of the file, where the EXIF block lives.
pub fn read_exif_gps(path: &Path) -> Option<GeodeticPosition> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = Vec::with_capacity(1 << 17);
    file.by_ref().take(1 << 17).read_to_end(&mut head).ok()?;
    exif_gps(&head)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal JPEG whose EXIF holds a GPS IFD.
    pub(super) fn jpeg_with_gps(
        little_endian: bool,
        lat: (u32, u32, u32, u32),
        lat_ref: u8,
        lon: (u32, u32, u32, u32),
        lon_ref: u8,
        alt: Option<(u32, u32, u8)>,
    ) -> Vec<u8> {
        let w16 = |v: u16| {
            if little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let w32 = |v: u32| {
            if little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let mut t: Vec<u8> = Vec::new();
        t.extend(if little_endian { b"II" } else { b"MM" });
        t.extend(w16(42));
        t.extend(w32(8));
        // IFD0 at 8: one entry (GPS IFD pointer), next-IFD 0.
        t.extend(w16(1));
        t.extend(w16(0x8825));
        t.extend(w16(4));
        t.extend(w32(1));
        t.extend(w32(26)); // GPS IFD offset
        t.extend(w32(0));
        assert_eq!(t.len(), 26);
        let n: u16 = if alt.is_some() { 6 } else { 4 };
        let data = 26 + 2 + 12 * n as usize + 4;
        let mut entries: Vec<u8> = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let ascii = |entries: &mut Vec<u8>, tag: u16, c: u8| {
            entries.extend(w16(tag));
            entries.extend(w16(2));
            entries.extend(w32(2));
            entries.extend([c, 0, 0, 0]);
        };
        let mut rational = |entries: &mut Vec<u8>, tag: u16, values: &[(u32, u32)]| {
            entries.extend(w16(tag));
            entries.extend(w16(5));
            entries.extend(w32(values.len() as u32));
            entries.extend(w32((data + blob.len()) as u32));
            for &(num, den) in values {
                blob.extend(w32(num));
                blob.extend(w32(den));
            }
        };
        ascii(&mut entries, 1, lat_ref);
        rational(&mut entries, 2, &[(lat.0, 1), (lat.1, 1), (lat.2, lat.3)]);
        ascii(&mut entries, 3, lon_ref);
        rational(&mut entries, 4, &[(lon.0, 1), (lon.1, 1), (lon.2, lon.3)]);
        if let Some((num, den, below)) = alt {
            entries.extend(w16(5));
            entries.extend(w16(1));
            entries.extend(w32(1));
            entries.extend([below, 0, 0, 0]);
            rational(&mut entries, 6, &[(num, den)]);
        }
        t.extend(w16(n));
        t.extend(entries);
        t.extend(w32(0));
        assert_eq!(t.len(), data);
        t.extend(blob);
        let mut jpeg = vec![0xFF, 0xD8];
        // An APP0 first, as real files have.
        jpeg.extend([0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00]);
        jpeg.extend([0xFF, 0xE1]);
        jpeg.extend(((t.len() + 8) as u16).to_be_bytes());
        jpeg.extend(b"Exif\0\0");
        jpeg.extend(t);
        jpeg.extend([0xFF, 0xDA, 0x00, 0x02, 0xFF, 0xD9]);
        jpeg
    }

    #[test]
    fn exif_gps_reads_both_byte_orders_refs_and_altitude() {
        // 35°40'52.45"N 139°46'1.65"E, 40.25 m.
        for le in [true, false] {
            let jpeg = jpeg_with_gps(
                le,
                (35, 40, 5245, 100),
                b'N',
                (139, 46, 165, 100),
                b'E',
                Some((4025, 100, 0)),
            );
            let fix = exif_gps(&jpeg).unwrap();
            assert!((fix.lat_deg - (35.0 + 40.0 / 60.0 + 52.45 / 3600.0)).abs() < 1e-12);
            assert!((fix.lon_deg - (139.0 + 46.0 / 60.0 + 1.65 / 3600.0)).abs() < 1e-12);
            assert_eq!(fix.alt_m, Some(40.25));
        }
        let south_west_below = jpeg_with_gps(
            true,
            (33, 52, 0, 1),
            b'S',
            (151, 12, 30, 1),
            b'W',
            Some((5, 1, 1)),
        );
        let fix = exif_gps(&south_west_below).unwrap();
        assert!(fix.lat_deg < -33.0 && fix.lon_deg < -151.0);
        assert_eq!(fix.alt_m, Some(-5.0));
        let no_alt = jpeg_with_gps(true, (1, 2, 3, 1), b'N', (4, 5, 6, 1), b'E', None);
        assert_eq!(exif_gps(&no_alt).unwrap().alt_m, None);
        let zero = jpeg_with_gps(true, (0, 0, 0, 1), b'N', (0, 0, 0, 1), b'E', None);
        assert_eq!(exif_gps(&zero), None);
        assert_eq!(exif_gps(b"not a jpeg"), None);
        assert_eq!(exif_gps(&[0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02]), None);
    }

    #[test]
    fn prior_file_parses_and_converts_to_enu() {
        let text = "\
# name lat lon alt std_h std_v
a.jpg, 35.0, 139.0, 10.0
sub/b.jpg 35.001 139.0 12.0 2.0
c 35.0 139.001 10.0 2.0 4.0
";
        let set = GpsPriorSet::parse(text, PriorCoordinates::Wgs84, "priors.txt").unwrap();
        assert_eq!(set.len(), 3);
        let names: Vec<String> = ["a.jpg", "b.jpg", "images/c.jpg", "d.jpg"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let origin = GeodeticPosition {
            lat_deg: 35.0,
            lon_deg: 139.0,
            alt_m: Some(10.0),
        };
        let (priors, frame) = set.to_local_enu(&names, Some(origin), Vector3::new(5.0, 5.0, 10.0));
        assert!(matches!(frame, PriorFrame::Enu(_)));
        let a = priors[0].unwrap();
        assert!(a.position.coords.norm() < 1e-6);
        assert_eq!(a.std_dev, Vector3::new(5.0, 5.0, 10.0));
        // 0.001° of latitude is ~110.9 m north.
        let b = priors[1].unwrap();
        assert!(b.position.x.abs() < 0.01);
        assert!((b.position.y - 110.95).abs() < 0.2, "{}", b.position.y);
        assert!((b.position.z - 2.0).abs() < 0.01);
        assert_eq!(b.std_dev, Vector3::new(2.0, 2.0, 2.0));
        // 0.001° of longitude at 35° N is ~91.3 m east.
        let c = priors[2].unwrap();
        assert!((c.position.x - 91.29).abs() < 0.2, "{}", c.position.x);
        assert_eq!(c.std_dev, Vector3::new(2.0, 2.0, 4.0));
        assert!(priors[3].is_none());
        assert!(frame.describe().contains("35.000000000 139.000000000"));
    }

    #[test]
    fn prior_file_errors_and_cartesian_and_missing_altitude() {
        assert!(GpsPriorSet::parse("a 1 2", PriorCoordinates::Wgs84, "p").is_err());
        assert!(GpsPriorSet::parse("a 91 2 3", PriorCoordinates::Wgs84, "p").is_err());
        assert!(GpsPriorSet::parse("a x 2 3", PriorCoordinates::Wgs84, "p").is_err());
        let set = GpsPriorSet::parse("a 100 -20 3\n", PriorCoordinates::Cartesian, "p").unwrap();
        let (priors, frame) =
            set.to_local_enu(&["a.png".to_string()], None, Vector3::new(1.0, 1.0, 2.0));
        assert_eq!(frame, PriorFrame::Cartesian);
        assert_eq!(priors[0].unwrap().position, Point3::new(100.0, -20.0, 3.0));

        let mut exif = GpsPriorSet::new(PriorCoordinates::Wgs84);
        exif.insert_geodetic(
            "x.jpg",
            GeodeticPosition {
                lat_deg: 10.0,
                lon_deg: 20.0,
                alt_m: None,
            },
        );
        let (priors, _) =
            exif.to_local_enu(&["x.jpg".to_string()], None, Vector3::new(5.0, 5.0, 10.0));
        let x = priors[0].unwrap();
        assert!(x.position.coords.norm() < 1e-6);
        assert!(x.std_dev.z.is_infinite());
    }
}
