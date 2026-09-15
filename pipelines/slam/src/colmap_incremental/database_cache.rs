//! Faithful (C1-scoped) port of `scene/database_cache.h`'s [`DatabaseCache`]
//! container, backed by this repository's own verified-graph export formats
//! instead of a COLMAP `database.db`.
//!
//! Ported surface, cited to `src/colmap/scene/database_cache.h` (fetched at
//! the commit pinned by `docs/colmap_rig_mapper_port_plan.md` §0):
//! - [`DatabaseCache`] itself: `NumRigs`/`NumCameras`/`NumFrames`/`NumImages`
//!   (`.h:86-89`), `AddRig`/`AddCamera`/`AddFrame`/`AddImage` (`.h:93-96`),
//!   the `Rig`/`Camera`/`Frame`/`Image` accessors and `Exists*` checks
//!   (`.h:100-120`), `CorrespondenceGraph()` (`.h:123-125`),
//!   `FindImageWithName` (`.h:128`).
//! - **Not ported**: `DatabaseCache::Load` itself (`database_cache.cc`,
//!   not fetched for this task — see §3.2 of the port plan and the task
//!   brief, which scoped C1's reading list to the `.h` files plus
//!   `database_cache.h`/`.cc` "for the types"). COLMAP's `Load` reads a
//!   SQLite `database.db`; this repository has no such database for the
//!   rig-mapper parity harness (`docs/colmap_rig_mapper_port_plan.md` §4.1)
//!   — the actual on-disk inputs are the `generalized-rig-manifest-v1` text
//!   manifest, `*_features.txt` keypoint dumps, and a COLMAP-frontend pair
//!   export, exactly as `examples/generalized_rig_sfm.rs` /
//!   `examples/import_colmap_verified_snapshot.rs` already consume for the
//!   M9 benchmark. [`DatabaseCache::from_generalized_rig_export`] is this
//!   module's equivalent entry point, replaying the *semantics* of `Load`
//!   (build rigs/cameras/frames/images, ingest verified pairs into the
//!   correspondence graph, `Finalize()` it) against those inputs instead.
//!
//! ## Reuse, and why the manifest/pair readers below are new code
//!
//! `docs/colmap_rig_mapper_port_plan.md` §3.2 asks this milestone to reuse
//! M2's persistent [`CorrespondenceGraph`] "verbatim" if it has landed — it
//! has (`crates/vision/src/two_view/correspondence_graph.rs`, a cited,
//! line-for-line port of `scene/correspondence_graph.h/.cc`), and this
//! module reuses it **directly, unmodified** (see the `use` below) rather
//! than reimplementing it. Its already-snake_cased methods are exactly
//! COLMAP's API subset this task asks for:
//! `num_correspondences_for_image` = `NumCorrespondencesForImage`,
//! `num_matches_between_images` = `NumCorrespondencesBetweenImages`,
//! `find_correspondences` = `FindCorrespondences`,
//! `extract_transitive_correspondences` = `FindTransitiveCorrespondences`,
//! `extract_correspondences` = `ExtractCorrespondences`,
//! `has_correspondences` = `HasCorrespondences`,
//! `is_two_view_observation` = `IsTwoViewObservation`.
//!
//! The task also asks to reuse the rig-manifest/features/verified-pair
//! *readers* already used by `examples/generalized_rig_sfm.rs` /
//! `examples/unordered_sfm_demo.rs` instead of writing new parsers. That
//! turns out to be **architecturally impossible without a workspace
//! refactor**: those readers (`examples/generalized_rig_sfm.rs`'s private
//! `parse_manifest`, and `src/verified_pair_snapshot.rs`'s checksummed
//! `.vps` codec) live in the root `visloc-rs` binary crate, which
//! **depends on** `visloc-slam` (`Cargo.toml:75`) — `pipelines/slam` cannot
//! depend back on it without introducing a workspace cycle. Moving those
//! readers into a shared lower crate would fix this properly but is a
//! multi-file refactor (13+ call sites across `examples/`) well outside
//! this milestone's scope; flagged as a follow-up.
//!
//! Given that constraint, this module reads the two *plain, trivial-format*
//! inputs directly (mirroring, not importing, the formats
//! `examples/generalized_rig_sfm.rs::parse_manifest` and the
//! `*_features.txt` convention already define): the
//! `generalized-rig-manifest-v1` text manifest ([`parse_rig_manifest`]) and
//! the `x y` per line keypoint dumps ([`parse_features_txt`]). For the
//! verified-pair graph it deliberately does **not** reimplement the
//! checksummed `.vps` codec (`src/verified_pair_snapshot.rs`, 1700+ lines,
//! FNV-1a integrity checks, raw-index bookkeeping) — instead it reads the
//! simpler upstream COLMAP-frontend export that `.vps` is itself built
//! from (`examples/import_colmap_verified_snapshot.rs`'s documented
//! `VISLOC-COLMAP-1` binary layout: magic, `(name, feature_count)` per
//! image, then `(image_i, image_j, [(u32, u32)...])` per pair —
//! [`parse_colmap_pairs_export`] mirrors that same fixed layout). Both
//! readers are small, direct transcriptions of an already-documented
//! on-disk contract, not new *parsing logic* in the sense of new format
//! design.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;

use nalgebra::{Point2, Quaternion, UnitQuaternion, Vector3};
use thiserror::Error;

use visloc_core::geometry::SE3;
use visloc_vision::two_view::{ConfigurationType, CorrespondenceGraph};

use super::reconstruction::{Camera, Image, Point2D};
use super::types::{CameraT, DataT, Frame, FrameT, ImageT, Rig, RigT, SensorT};

#[derive(Debug, Error)]
pub enum DatabaseCacheError {
    #[error("read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {message}")]
    Parse { path: String, message: String },
    #[error("inconsistent input: {0}")]
    Inconsistent(String),
}

/// Port of `class DatabaseCache` (`scene/database_cache.h:49-139`),
/// minus `PosePrior` bookkeeping (`.h:97,114,90,131,137`, `ConvertPosePriorsToENU`)
/// — this module's inputs carry no pose priors.
#[derive(Debug, Clone, Default)]
pub struct DatabaseCache {
    rigs: BTreeMap<RigT, Rig>,
    cameras: BTreeMap<CameraT, Camera>,
    frames: BTreeMap<FrameT, Frame>,
    images: BTreeMap<ImageT, Image>,
    correspondence_graph: CorrespondenceGraph,
}

impl DatabaseCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `DatabaseCache` directly from its constituent parts (e.g. a
    /// synthetic scene, or a future non-file-based loader) rather than the
    /// `generalized-rig-manifest-v1`/features/pairs-export file triple.
    pub fn from_parts(
        rigs: BTreeMap<RigT, Rig>,
        cameras: BTreeMap<CameraT, Camera>,
        frames: BTreeMap<FrameT, Frame>,
        images: BTreeMap<ImageT, Image>,
        correspondence_graph: CorrespondenceGraph,
    ) -> Self {
        Self {
            rigs,
            cameras,
            frames,
            images,
            correspondence_graph,
        }
    }

    pub fn num_rigs(&self) -> usize {
        self.rigs.len()
    }
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    pub fn add_rig(&mut self, rig: Rig) {
        let rig_id = rig.rig_id();
        let previous = self.rigs.insert(rig_id, rig);
        assert!(previous.is_none(), "rig {rig_id} already exists");
    }
    pub fn add_camera(&mut self, camera: Camera) {
        let camera_id = camera.id;
        let previous = self.cameras.insert(camera_id, camera);
        assert!(previous.is_none(), "camera {camera_id} already exists");
    }
    pub fn add_frame(&mut self, frame: Frame) {
        let frame_id = frame.frame_id();
        let previous = self.frames.insert(frame_id, frame);
        assert!(previous.is_none(), "frame {frame_id} already exists");
    }
    pub fn add_image(&mut self, image: Image) {
        let image_id = image.image_id;
        let previous = self.images.insert(image_id, image);
        assert!(previous.is_none(), "image {image_id} already exists");
    }

    pub fn rig(&self, rig_id: RigT) -> &Rig {
        self.rigs
            .get(&rig_id)
            .unwrap_or_else(|| panic!("rig {rig_id} does not exist"))
    }
    pub fn camera(&self, camera_id: CameraT) -> &Camera {
        self.cameras
            .get(&camera_id)
            .unwrap_or_else(|| panic!("camera {camera_id} does not exist"))
    }
    pub fn frame(&self, frame_id: FrameT) -> &Frame {
        self.frames
            .get(&frame_id)
            .unwrap_or_else(|| panic!("frame {frame_id} does not exist"))
    }
    pub fn image(&self, image_id: ImageT) -> &Image {
        self.images
            .get(&image_id)
            .unwrap_or_else(|| panic!("image {image_id} does not exist"))
    }

    pub fn rigs(&self) -> &BTreeMap<RigT, Rig> {
        &self.rigs
    }
    pub fn cameras(&self) -> &BTreeMap<CameraT, Camera> {
        &self.cameras
    }
    pub fn frames(&self) -> &BTreeMap<FrameT, Frame> {
        &self.frames
    }
    pub fn images(&self) -> &BTreeMap<ImageT, Image> {
        &self.images
    }

    pub fn exists_rig(&self, rig_id: RigT) -> bool {
        self.rigs.contains_key(&rig_id)
    }
    pub fn exists_camera(&self, camera_id: CameraT) -> bool {
        self.cameras.contains_key(&camera_id)
    }
    pub fn exists_frame(&self, frame_id: FrameT) -> bool {
        self.frames.contains_key(&frame_id)
    }
    pub fn exists_image(&self, image_id: ImageT) -> bool {
        self.images.contains_key(&image_id)
    }

    /// Port of `CorrespondenceGraph()` (`.h:123-125`).
    pub fn correspondence_graph(&self) -> &CorrespondenceGraph {
        &self.correspondence_graph
    }
    pub fn correspondence_graph_mut(&mut self) -> &mut CorrespondenceGraph {
        &mut self.correspondence_graph
    }

    /// Port of `FindImageWithName` (`.h:128`): linear search, exactly as
    /// COLMAP documents.
    pub fn find_image_with_name(&self, name: &str) -> Option<&Image> {
        self.images.values().find(|image| image.name == name)
    }

    /// Build a `DatabaseCache` from the visloc generalized-rig export triple
    /// this repository's mapper harnesses already consume: a
    /// `generalized-rig-manifest-v1` text manifest (rig sensors + frame
    /// membership), a `*_features.txt` keypoint dump per image, and a
    /// COLMAP-frontend verified-pair export (`VISLOC-COLMAP-1` layout, see
    /// module doc). Mirrors `DatabaseCache::Load`'s effect (build the object
    /// graph, ingest every verified pair into the correspondence graph,
    /// `Finalize()` it) against these inputs instead of a `database.db`.
    ///
    /// The lowest-indexed manifest sensor (`S 0 ...`) is treated as the
    /// rig's reference sensor — the OpenLORIS manifests this module was
    /// built against always give sensor 0 an identity `sensor_from_rig`
    /// (`openloris-tier1000-rig-manifest-v1.txt`'s `S 0 1 848 800 ... 1 0 0 0
    /// 0 0 0`), matching that convention; a manifest that violated it would
    /// silently get a non-identity "reference" pose composed into every
    /// frame instead — flagged as a documented assumption, not enforced by
    /// an assertion, since the manifest format itself has no explicit
    /// reference-sensor marker to check against.
    pub fn from_generalized_rig_export(
        manifest_path: &Path,
        features_dir: &Path,
        pairs_export_path: &Path,
    ) -> Result<Self, DatabaseCacheError> {
        let manifest = parse_rig_manifest(manifest_path)?;
        let pairs_export = parse_colmap_pairs_export(pairs_export_path)?;

        // name -> (frame_idx, sensor_index), from the manifest's F rows.
        let mut frame_of_name: HashMap<&str, (u64, usize)> =
            HashMap::with_capacity(manifest.frame_rows.len());
        for row in &manifest.frame_rows {
            frame_of_name.insert(row.name.as_str(), (row.frame_idx, row.sensor_index));
        }

        let mut cache = DatabaseCache::new();

        let mut rig = Rig::new();
        rig.set_rig_id(0);
        let ref_index = manifest
            .sensors
            .iter()
            .map(|s| s.index)
            .min()
            .ok_or_else(|| DatabaseCacheError::Inconsistent("manifest has no sensors".into()))?;
        for sensor in &manifest.sensors {
            let sensor_id = SensorT::camera(sensor.camera_id);
            if sensor.index == ref_index {
                rig.add_ref_sensor(sensor_id);
            } else {
                rig.add_sensor(sensor_id, Some(sensor.sensor_from_rig.clone()));
            }
            cache.add_camera(sensor.camera.clone());
        }
        cache.add_rig(rig);

        let sensor_by_index: HashMap<usize, CameraT> = manifest
            .sensors
            .iter()
            .map(|s| (s.index, s.camera_id))
            .collect();

        // Build frames (one per distinct frame_idx) lazily as images are
        // walked, in pairs_export's own image order — that order becomes
        // this cache's `image_t` numbering (0-based), avoiding any need to
        // reconcile it against the manifest's own row order.
        let mut frame_ids_seen: std::collections::BTreeSet<FrameT> =
            std::collections::BTreeSet::new();

        for (image_index, name) in pairs_export.image_names.iter().enumerate() {
            let image_id = image_index as ImageT;
            let &(frame_idx, sensor_index) = frame_of_name.get(name.as_str()).ok_or_else(|| {
                DatabaseCacheError::Inconsistent(format!(
                    "pair export image {name:?} is absent from the rig manifest"
                ))
            })?;
            let &camera_id = sensor_by_index.get(&sensor_index).ok_or_else(|| {
                DatabaseCacheError::Inconsistent(format!(
                    "manifest frame row for {name:?} references unknown sensor index {sensor_index}"
                ))
            })?;

            if frame_ids_seen.insert(frame_idx) {
                cache.add_frame(Frame::new(frame_idx, 0));
            }
            cache
                .frames
                .get_mut(&frame_idx)
                .unwrap()
                .add_data_id(DataT::camera(camera_id, image_id));

            let features_path = features_dir.join(format!("{}_features.txt", stem(name)));
            let keypoints = parse_features_txt(&features_path)?;
            let expected = pairs_export.feature_counts[image_index] as usize;
            if keypoints.len() != expected {
                return Err(DatabaseCacheError::Inconsistent(format!(
                    "{name}: pair export declares {expected} keypoints but {} reads {}",
                    features_path.display(),
                    keypoints.len()
                )));
            }

            let mut image = Image::new(image_id, camera_id, frame_idx, name.clone());
            image.points2d = keypoints.into_iter().map(Point2D::new).collect();
            cache.add_image(image);
        }

        // Correspondence graph: `AddImage` capacities first (COLMAP requires
        // every referenced image to be declared before
        // `AddTwoViewGeometry`, `correspondence_graph.cc:111-112`), then
        // ingest every verified pair, then `Finalize()`.
        for image_id in 0..pairs_export.image_names.len() as ImageT {
            let num_points2d = cache.image(image_id).num_points2d();
            cache
                .correspondence_graph
                .add_image(image_id as usize, num_points2d);
        }
        for pair in &pairs_export.pairs {
            cache
                .correspondence_graph
                .add_two_view_geometry(
                    pair.image_i,
                    pair.image_j,
                    &pair.matches,
                    ConfigurationType::Calibrated,
                )
                .map_err(|error| {
                    DatabaseCacheError::Inconsistent(format!(
                        "ingesting pair ({}, {}): {error}",
                        pair.image_i, pair.image_j
                    ))
                })?;
        }
        cache.correspondence_graph.finalize();

        Ok(cache)
    }
}

fn stem(file_name: &str) -> &str {
    match file_name.rfind('.') {
        Some(dot) => &file_name[..dot],
        None => file_name,
    }
}

// ---------------------------------------------------------------------
// `generalized-rig-manifest-v1` text manifest reader.
// ---------------------------------------------------------------------

struct ManifestSensor {
    index: usize,
    camera_id: CameraT,
    camera: Camera,
    /// Identity for the (by-convention) reference sensor; unused for it.
    sensor_from_rig: SE3,
}

struct ManifestFrameRow {
    frame_idx: u64,
    name: String,
    sensor_index: usize,
}

struct ParsedRigManifest {
    sensors: Vec<ManifestSensor>,
    frame_rows: Vec<ManifestFrameRow>,
}

/// Reads the `generalized-rig-manifest-v1` text format: `S index camera_id
/// width height fx fy cx cy qw qx qy qz tx ty tz` sensor rows and `F
/// frame_idx image_name sensor_index` frame-membership rows, exactly the
/// contract `examples/generalized_rig_sfm.rs::parse_manifest` reads (see
/// module doc for why that function itself can't be called from here).
fn parse_rig_manifest(path: &Path) -> Result<ParsedRigManifest, DatabaseCacheError> {
    let contents = fs::read_to_string(path).map_err(|source| DatabaseCacheError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut sensors = Vec::new();
    let mut frame_rows = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let line = zero_line + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let err = |message: String| DatabaseCacheError::Parse {
            path: path.display().to_string(),
            message: format!("line {line}: {message}"),
        };
        match tokens.first().copied() {
            Some("S") => {
                if tokens.len() != 16 {
                    return Err(err(format!(
                        "sensor row requires 16 fields, found {}",
                        tokens.len()
                    )));
                }
                let index: usize = parse_field(&tokens, 1, &err)?;
                let camera_id: CameraT = parse_field(&tokens, 2, &err)?;
                let width: u32 = parse_field(&tokens, 3, &err)?;
                let height: u32 = parse_field(&tokens, 4, &err)?;
                let fx: f64 = parse_field(&tokens, 5, &err)?;
                let fy: f64 = parse_field(&tokens, 6, &err)?;
                let cx: f64 = parse_field(&tokens, 7, &err)?;
                let cy: f64 = parse_field(&tokens, 8, &err)?;
                let qw: f64 = parse_field(&tokens, 9, &err)?;
                let qx: f64 = parse_field(&tokens, 10, &err)?;
                let qy: f64 = parse_field(&tokens, 11, &err)?;
                let qz: f64 = parse_field(&tokens, 12, &err)?;
                let tx: f64 = parse_field(&tokens, 13, &err)?;
                let ty: f64 = parse_field(&tokens, 14, &err)?;
                let tz: f64 = parse_field(&tokens, 15, &err)?;
                let quaternion = Quaternion::new(qw, qx, qy, qz);
                if !quaternion.norm().is_finite() || quaternion.norm() <= 1.0e-12 {
                    return Err(err("invalid sensor quaternion".into()));
                }
                sensors.push(ManifestSensor {
                    index,
                    camera_id,
                    camera: Camera::pinhole(camera_id, width, height, fx, fy, cx, cy),
                    sensor_from_rig: SE3::new(
                        UnitQuaternion::new_normalize(quaternion),
                        Vector3::new(tx, ty, tz),
                    ),
                });
            }
            Some("F") => {
                if tokens.len() != 4 {
                    return Err(err(format!(
                        "frame row requires 4 fields, found {}",
                        tokens.len()
                    )));
                }
                frame_rows.push(ManifestFrameRow {
                    frame_idx: parse_field(&tokens, 1, &err)?,
                    name: tokens[2].to_owned(),
                    sensor_index: parse_field(&tokens, 3, &err)?,
                });
            }
            Some(kind) => return Err(err(format!("unknown row kind {kind}"))),
            None => {}
        }
    }
    if sensors.len() < 2 || frame_rows.is_empty() {
        return Err(DatabaseCacheError::Parse {
            path: path.display().to_string(),
            message: "manifest requires at least two sensors and one frame row".into(),
        });
    }
    Ok(ParsedRigManifest {
        sensors,
        frame_rows,
    })
}

fn parse_field<T: std::str::FromStr>(
    tokens: &[&str],
    index: usize,
    err: &impl Fn(String) -> DatabaseCacheError,
) -> Result<T, DatabaseCacheError> {
    tokens
        .get(index)
        .ok_or_else(|| err(format!("missing field {index}")))?
        .parse::<T>()
        .map_err(|_| err(format!("field {index} is not a valid number")))
}

/// Reads a `*_features.txt` keypoint dump: one `x y` pair per line, one
/// line per detected keypoint (this repository's plain feature-export
/// convention, e.g.
/// `openloris-tier1000-colmap-graph-visloc-mapper-v1/export/features/*.txt`).
fn parse_features_txt(path: &Path) -> Result<Vec<Point2<f64>>, DatabaseCacheError> {
    let contents = fs::read_to_string(path).map_err(|source| DatabaseCacheError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut keypoints = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let text = raw.trim();
        if text.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() != 2 {
            return Err(DatabaseCacheError::Parse {
                path: path.display().to_string(),
                message: format!(
                    "line {}: expected 2 fields, found {}",
                    zero_line + 1,
                    tokens.len()
                ),
            });
        }
        let x: f64 = tokens[0].parse().map_err(|_| DatabaseCacheError::Parse {
            path: path.display().to_string(),
            message: format!("line {}: invalid x", zero_line + 1),
        })?;
        let y: f64 = tokens[1].parse().map_err(|_| DatabaseCacheError::Parse {
            path: path.display().to_string(),
            message: format!("line {}: invalid y", zero_line + 1),
        })?;
        keypoints.push(Point2::new(x, y));
    }
    Ok(keypoints)
}

// ---------------------------------------------------------------------
// COLMAP-frontend verified-pair export reader (`VISLOC-COLMAP-1` layout).
// ---------------------------------------------------------------------

const PAIRS_EXPORT_MAGIC: &[u8; 16] = b"VISLOC-COLMAP-1\0";

struct ColmapPairsExport {
    image_names: Vec<String>,
    feature_counts: Vec<u64>,
    pairs: Vec<ColmapPairRecord>,
}

struct ColmapPairRecord {
    image_i: usize,
    image_j: usize,
    matches: Vec<(usize, usize)>,
}

/// Reads the fixed binary layout `examples/import_colmap_verified_snapshot.rs`
/// documents and consumes (magic `VISLOC-COLMAP-1\0`; image_count then
/// `(name, feature_count)` per image; pair_count then `(image_i, image_j,
/// match_count, [(u32, u32)...])` per pair) — see module doc for why this is
/// read directly rather than through the checksummed `.vps` codec it is
/// itself the source of.
fn parse_colmap_pairs_export(path: &Path) -> Result<ColmapPairsExport, DatabaseCacheError> {
    let file = fs::File::open(path).map_err(|source| DatabaseCacheError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let err = |message: String| DatabaseCacheError::Parse {
        path: path.display().to_string(),
        message,
    };

    let mut magic = [0u8; 16];
    reader
        .read_exact(&mut magic)
        .map_err(|source| DatabaseCacheError::Io {
            path: path.display().to_string(),
            source,
        })?;
    if &magic != PAIRS_EXPORT_MAGIC {
        return Err(err("unrecognized magic (expected VISLOC-COLMAP-1)".into()));
    }

    let image_count = read_u64(&mut reader, &err)? as usize;
    let mut image_names = Vec::with_capacity(image_count);
    let mut feature_counts = Vec::with_capacity(image_count);
    for _ in 0..image_count {
        image_names.push(read_string(&mut reader, &err)?);
        feature_counts.push(read_u64(&mut reader, &err)?);
    }

    let pair_count = read_u64(&mut reader, &err)? as usize;
    let mut pairs = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        let image_i = read_u64(&mut reader, &err)? as usize;
        let image_j = read_u64(&mut reader, &err)? as usize;
        let match_count = read_u64(&mut reader, &err)? as usize;
        let mut matches = Vec::with_capacity(match_count);
        for _ in 0..match_count {
            let mut bytes = [0u8; 8];
            reader
                .read_exact(&mut bytes)
                .map_err(|source| DatabaseCacheError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
            let left = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
            let right = u32::from_le_bytes(bytes[4..].try_into().unwrap()) as usize;
            matches.push((left, right));
        }
        pairs.push(ColmapPairRecord {
            image_i,
            image_j,
            matches,
        });
    }

    Ok(ColmapPairsExport {
        image_names,
        feature_counts,
        pairs,
    })
}

fn read_u64(
    reader: &mut impl Read,
    err: &impl Fn(String) -> DatabaseCacheError,
) -> Result<u64, DatabaseCacheError> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| err("unexpected end of file reading u64".into()))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string(
    reader: &mut impl Read,
    err: &impl Fn(String) -> DatabaseCacheError,
) -> Result<String, DatabaseCacheError> {
    let length = read_u64(reader, err)? as usize;
    let mut bytes = vec![0u8; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| err("unexpected end of file reading string".into()))?;
    String::from_utf8(bytes).map_err(|_| err("name is not valid UTF-8".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn synthetic_manifest_and_export(
        dir: &Path,
    ) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        // Two-sensor rig, two frames (four images), a chain of pairwise
        // matches touching every image at least once.
        let manifest_path = dir.join("manifest.txt");
        fs::write(
            &manifest_path,
            "# generalized-rig-manifest-v1\n\
             S 0 1 640 480 500 500 320 240 1 0 0 0 0 0 0\n\
             S 1 2 640 480 500 500 320 240 1 0 0 0 -0.1 0 0\n\
             F 0 cam1_0.png 0\n\
             F 0 cam2_0.png 1\n\
             F 1 cam1_1.png 0\n\
             F 1 cam2_1.png 1\n",
        )
        .unwrap();

        let features_dir = dir.join("features");
        fs::create_dir_all(&features_dir).unwrap();
        for (name, count) in [("cam1_0", 3), ("cam2_0", 3), ("cam1_1", 2), ("cam2_1", 2)] {
            let mut text = String::new();
            for k in 0..count {
                text.push_str(&format!("{}.0 {}.0\n", 10 + k, 20 + k));
            }
            fs::write(features_dir.join(format!("{name}_features.txt")), text).unwrap();
        }

        let pairs_path = dir.join("pairs.bin");
        let mut file = fs::File::create(&pairs_path).unwrap();
        file.write_all(PAIRS_EXPORT_MAGIC).unwrap();
        let names = ["cam1_0.png", "cam2_0.png", "cam1_1.png", "cam2_1.png"];
        let counts = [3u64, 3, 2, 2];
        file.write_all(&(names.len() as u64).to_le_bytes()).unwrap();
        for (name, count) in names.iter().zip(counts.iter()) {
            file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            file.write_all(name.as_bytes()).unwrap();
            file.write_all(&count.to_le_bytes()).unwrap();
        }
        // Pairs: (0,1) same-frame stereo, (0,2)/(1,3) cross-time same-sensor
        // — every image gets at least one correspondence so none are
        // dropped by `CorrespondenceGraph::finalize()`'s documented
        // zero-observation-image contract (see the module doc on
        // `crates/vision/src/two_view/correspondence_graph.rs`).
        let pairs: Vec<(u64, u64, Vec<(u32, u32)>)> = vec![
            (0, 1, vec![(0, 0), (1, 1)]),
            (0, 2, vec![(2, 0)]),
            (1, 3, vec![(2, 0)]),
        ];
        file.write_all(&(pairs.len() as u64).to_le_bytes()).unwrap();
        for (i, j, matches) in &pairs {
            file.write_all(&i.to_le_bytes()).unwrap();
            file.write_all(&j.to_le_bytes()).unwrap();
            file.write_all(&(matches.len() as u64).to_le_bytes())
                .unwrap();
            for (l, r) in matches {
                file.write_all(&l.to_le_bytes()).unwrap();
                file.write_all(&r.to_le_bytes()).unwrap();
            }
        }
        drop(file);

        (manifest_path, features_dir, pairs_path)
    }

    #[test]
    fn from_generalized_rig_export_builds_expected_object_graph() {
        let dir = std::env::temp_dir().join(format!(
            "colmap_incremental_dbcache_test_{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let (manifest_path, features_dir, pairs_path) = synthetic_manifest_and_export(&dir);

        let cache =
            DatabaseCache::from_generalized_rig_export(&manifest_path, &features_dir, &pairs_path)
                .unwrap();

        assert_eq!(cache.num_rigs(), 1);
        assert_eq!(cache.num_cameras(), 2);
        assert_eq!(cache.num_frames(), 2);
        assert_eq!(cache.num_images(), 4);
        assert!(cache.rig(0).is_ref_sensor(SensorT::camera(1)));
        assert!(!cache.rig(0).is_ref_sensor(SensorT::camera(2)));

        assert_eq!(cache.image(0).name, "cam1_0.png");
        assert_eq!(cache.image(0).frame_id, 0);
        assert_eq!(cache.image(1).frame_id, 0);
        assert_eq!(cache.image(2).frame_id, 1);
        assert_eq!(cache.frame(0).num_data_ids(), 2);

        let graph = cache.correspondence_graph();
        assert_eq!(graph.num_image_pairs(), 3);
        // 2 matches on (0,1) + 1 match on (0,2) + 1 match on (1,3) = 4
        // matches, each ingested bidirectionally -> sum of per-image counts
        // is 2 * 4 = 8.
        let total: usize = (0..cache.num_images())
            .map(|id| graph.num_correspondences_for_image(id))
            .sum();
        assert_eq!(total, 8);
        assert!(graph.has_correspondences(0, 0));
        assert_eq!(graph.find_correspondences(0, 0).len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_manifest_image_is_rejected() {
        let dir = std::env::temp_dir().join(format!(
            "colmap_incremental_dbcache_missing_test_{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let (manifest_path, features_dir, _pairs_path) = synthetic_manifest_and_export(&dir);

        // A pairs export that references an image absent from the manifest.
        let pairs_path = dir.join("pairs_bad.bin");
        let mut file = fs::File::create(&pairs_path).unwrap();
        file.write_all(PAIRS_EXPORT_MAGIC).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        let name = "unknown.png";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name.as_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // pair_count = 0
        drop(file);

        let result =
            DatabaseCache::from_generalized_rig_export(&manifest_path, &features_dir, &pairs_path);
        assert!(matches!(result, Err(DatabaseCacheError::Inconsistent(_))));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Real-1k-input integration test (`docs/colmap_rig_mapper_port_plan.md`
    /// §4.1's inputs). Skips gracefully — rather than failing — when the
    /// dataset is not present on this machine, since it is not checked into
    /// the repository (`No dataset writes` / read-only fixture).
    #[test]
    fn real_1k_tier_loads_expected_counts_if_present() {
        let dataset_root =
            Path::new("/home/sasaki/datasets/openloris/m9-learned-retrieval-models-v1");
        let manifest_path = dataset_root.join("openloris-tier1000-rig-manifest-v1.txt");
        let export_dir =
            dataset_root.join("openloris-tier1000-colmap-graph-visloc-mapper-v1/export");
        let features_dir = export_dir.join("features");
        let pairs_path = export_dir.join("pairs.bin");

        if !manifest_path.exists() || !features_dir.is_dir() || !pairs_path.exists() {
            eprintln!(
                "skipping real_1k_tier_loads_expected_counts_if_present: dataset not present at {}",
                dataset_root.display()
            );
            return;
        }

        let cache =
            DatabaseCache::from_generalized_rig_export(&manifest_path, &features_dir, &pairs_path)
                .expect("failed to load the real 1k tier export");

        assert_eq!(cache.num_images(), 1000);
        assert_eq!(cache.num_frames(), 500);

        let expected_pairs = {
            let export = parse_colmap_pairs_export(&pairs_path).unwrap();
            export.pairs.len()
        };
        assert_eq!(
            cache.correspondence_graph().num_image_pairs(),
            expected_pairs
        );

        let total_matches: usize = {
            let export = parse_colmap_pairs_export(&pairs_path).unwrap();
            export.pairs.iter().map(|pair| pair.matches.len()).sum()
        };
        let graph = cache.correspondence_graph();
        let sum_per_image: usize = (0..cache.num_images())
            .map(|id| graph.num_correspondences_for_image(id))
            .sum();
        assert_eq!(sum_per_image, 2 * total_matches);
    }
}
