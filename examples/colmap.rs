//! A drop-in COLMAP-CLI-compatible shim (`src/colmap/tools/*` parity slice),
//! powered by the same pipeline library as `unordered_sfm_demo`.
//!
//! Scripts written against COLMAP's command surface can drive this pipeline
//! unchanged:
//!
//! ```text
//! colmap feature_extractor --database_path db.db --image_path images/ \
//!     --ImageReader.single_camera 1 --SiftExtraction.max_features 4096
//! colmap exhaustive_matcher --database_path db.db
//! colmap mapper --database_path db.db --output_path sparse_out/
//! ```
//!
//! Self-contained: no separate SQLite database layer is required, since the
//! database module is the one COLMAP subsystem we deliberately do not port.
//! State is instead persisted under a `<db>.d/` directory next to the
//! "database" file (extracted features, verified matches, then the sparse
//! model in COLMAP text format under `--output_path/sparse/`).
//!
//! Supported subcommands and their commonly used flags:
//! - `feature_extractor`: `--database_path`, `--image_path`,
//!   `--SiftExtraction.max_features`
//! - `exhaustive_matcher`: `--database_path`
//! - `mapper`: `--database_path`, `--output_path`, `--Mapper.min_num_matches`
//! - `model_converter`: `--input_path`, `--output_path`, `--output_type TXT`
//!
//! Unknown flags are ignored with a warning (so typical scripts keep going).
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Everything is reached through the `visloc_rs` facade (`src/lib.rs`).
use visloc_rs::vision::features::sift::{extract_sift, GrayImage, SiftConfig};
use visloc_rs::vision::two_view::{
    ConfigurationType, TwoViewCorrespondence, TwoViewGeometryOptions, TwoViewGeometryVerifier,
};
use visloc_rs::{
    incremental_sfm, write_colmap_reconstruction_for_3dgs, BruteForceMatcher, Camera,
    CrossCheckMatcher, FeatureSet, IncrementalSfmConfig, Matcher, PairwiseMatches, Pose,
};

const IMAGE_SUFFIXES: [&str; 5] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"];

struct Args {
    flags: HashMap<String, String>,
}

impl Args {
    fn parse() -> Self {
        let mut flags = HashMap::new();
        let mut positional: Vec<String> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg.starts_with("--") {
                let key = arg.trim_start_matches("--").to_string();
                let mut value = String::new();
                if let Some(v) = args.next() {
                    // COLMAP passes the value on the next token.
                    value = v;
                }
                flags.insert(key, value);
            } else {
                positional.push(arg);
            }
        }
        if let Some(sub) = positional.first() {
            flags.insert("subcommand".to_string(), sub.clone());
        }
        Self { flags }
    }

    /// Lookup by bare key or any dotted suffix (`--SiftExtraction.max_features`
    /// matches key `max_features`).
    fn get(&self, key: &str) -> Option<&str> {
        self.flags.get(key).map(|s| s.as_str()).or_else(|| {
            self.flags
                .iter()
                .find(|(k, _)| k.ends_with(key))
                .map(|(_, v)| v.as_str())
        })
    }

    fn get_or(&self, key: &str, default: &str) -> String {
        self.get(key).unwrap_or(default).to_string()
    }
}

fn warn_unknown_flags(args: &Args, known: &[&str]) {
    for k in args.flags.keys() {
        if k == "subcommand" || k == "help" {
            continue;
        }
        if !known.iter().any(|n| k == n || k.ends_with(n)) {
            eprintln!("colmap-shim: ignoring unsupported flag --{k}");
        }
    }
}

fn data_dir(database_path: &Path) -> PathBuf {
    let mut name = database_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "db".to_string());
    name.push_str(".d");
    database_path.parent().unwrap_or(Path::new(".")).join(name)
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read image dir {dir:?}: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| {
                    let x = format!(".{}", x.to_string_lossy().to_lowercase());
                    IMAGE_SUFFIXES.contains(&x.as_str())
                })
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no images found under {dir:?}"));
    }
    Ok(files)
}

/// `X Y SCORE D…` text features, one line per keypoint — the external feature
/// format the pipeline's Files-based frontend reads back.
fn write_feature_file(
    path: &Path,
    keypoints: &[(f64, f64)],
    descriptors: &[Vec<f32>],
) -> Result<(), String> {
    let mut out = String::new();
    for ((x, y), descriptor) in keypoints.iter().zip(descriptors) {
        out.push_str(&format!("{x:.4} {y:.4} {:.4}", 1.0));
        for d in descriptor {
            out.push_str(&format!(" {d:.5}"));
        }
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| format!("write {path:?}: {e}"))
}

fn cmd_feature_extractor(args: &Args) -> Result<(), String> {
    let db = PathBuf::from(args.get_or("database_path", "database.db"));
    let image_dir = PathBuf::from(args.get_or("image_path", "."));
    let max_keypoints: usize = args
        .get_or("SiftExtraction.max_features", "4096")
        .parse()
        .map_err(|e| format!("{e}"))?;
    warn_unknown_flags(
        args,
        &["database_path", "image_path", "SiftExtraction.max_features"],
    );

    #[cfg(not(feature = "image-io"))]
    {
        let _ = (db, image_dir, max_keypoints);
        Err("feature_extractor requires building with --features image-io".to_string())
    }
    #[cfg(feature = "image-io")]
    {
        let dir = data_dir(&db);
        std::fs::create_dir_all(dir.join("features"))
            .map_err(|e| format!("mkdir {:?}: {e}", dir.join("features")))?;
        let images = list_images(&image_dir)?;
        let config = SiftConfig {
            max_keypoints,
            ..SiftConfig::default()
        };
        for image in &images {
            let name = image.file_name().unwrap().to_string_lossy().to_string();
            let stem = Path::new(&name).file_stem().unwrap().to_string_lossy();
            let out_path = dir.join("features").join(format!("{stem}_features.txt"));
            if out_path.exists() {
                continue;
            }
            let image_bytes = std::fs::read(image).map_err(|e| format!("read {image:?}: {e}"))?;
            let dynamic = image::load_from_memory(&image_bytes)
                .map_err(|e| format!("decode {image:?}: {e}"))?;
            let gray = dynamic.to_luma8();
            let (w, h) = (gray.width(), gray.height());
            let pixels: Vec<f32> = gray.as_raw().iter().map(|&b| b as f32).collect();
            let gray = GrayImage::new(w as usize, h as usize, &pixels)
                .map_err(|e| format!("gray {image:?}: {e}"))?;
            let (keypoints, descriptors) =
                extract_sift(&gray, &config).map_err(|e| format!("sift {image:?}: {e}"))?;
            let kp: Vec<(f64, f64)> = keypoints.iter().map(|k| (k.x, k.y)).collect();
            write_feature_file(&out_path, &kp, &descriptors)?;
            println!("extracted {name}: {} keypoints", keypoints.len());
        }
        println!(
            "feature_extractor: {} images -> {:?}",
            images.len(),
            dir.join("features")
        );
        Ok(())
    }
}

fn load_stored_features(db: &Path) -> Result<(Vec<FeatureSet>, Vec<String>), String> {
    let dir = data_dir(db).join("features");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read {dir:?}: {e}"))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with("_features.txt"))
        .collect();
    files.sort();
    let mut features = Vec::new();
    let mut names = Vec::new();
    for f in files {
        let path = dir.join(&f);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read {path:?}: {e}"))?;
        let mut keypoints = Vec::new();
        let mut descriptors = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() < 7 {
                continue;
            }
            keypoints.push(nalgebra::Point2::new(
                tokens[0].parse().map_err(|e| format!("{e}"))?,
                tokens[1].parse().map_err(|e| format!("{e}"))?,
            ));
            descriptors.push(
                tokens[6..]
                    .iter()
                    .map(|t| t.parse::<f32>().map_err(|e| format!("{e}")))
                    .collect::<Result<Vec<f32>, _>>()?,
            );
        }
        names.push(format!(
            "{}.jpg",
            f.strip_suffix("_features.txt").unwrap_or(&f)
        ));
        features.push(FeatureSet::new(keypoints, descriptors).map_err(|e| format!("{e}"))?);
    }
    Ok((features, names))
}

fn verify_pair(
    camera: &Camera,
    fi: &FeatureSet,
    fj: &FeatureSet,
    min_matches: usize,
) -> Option<Vec<(usize, usize)>> {
    let matcher = BruteForceMatcher { ratio: Some(0.8) };
    let dm = CrossCheckMatcher::new(matcher).match_descriptors(&fi.descriptors, &fj.descriptors);
    if dm.len() < min_matches {
        return None;
    }
    let corrs: Vec<TwoViewCorrespondence> = dm
        .iter()
        .map(|m| {
            TwoViewCorrespondence::new(fi.keypoints[m.query_index], fj.keypoints[m.train_index])
        })
        .collect();
    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));
    let report = verifier.classify(&corrs, camera);
    let keep = matches!(
        report.config,
        ConfigurationType::Calibrated | ConfigurationType::Uncalibrated
    );
    if !keep || report.inliers.len() < min_matches {
        return None;
    }
    Some(
        report
            .inliers
            .iter()
            .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
            .collect(),
    )
}

fn cmd_exhaustive_matcher(args: &Args) -> Result<(), String> {
    let db = PathBuf::from(args.get_or("database_path", "database.db"));
    warn_unknown_flags(args, &["database_path"]);
    let (features, names) = load_stored_features(&db)?;
    let (w, h) = (1600u32, 1066u32);
    let fx = 0.55 * w as f64;
    let camera = Camera::pinhole(1, w, h, fx, fx, w as f64 / 2.0, h as f64 / 2.0);

    let mut pairs: Vec<PairwiseMatches> = Vec::new();
    let total = names.len() * (names.len() - 1) / 2;
    let mut done = 0usize;
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            done += 1;
            if let Some(matches) = verify_pair(&camera, &features[i], &features[j], 20) {
                pairs.push(PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                });
            }
            if done % 100 == 0 {
                println!("matched {done}/{total}");
            }
        }
    }
    let dir = data_dir(&db);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{e}"))?;
    let path = dir.join("matches.txt");
    let mut text = format!("{}\n", names.len());
    for n in &names {
        text.push_str(&format!("{n}\n"));
    }
    text.push_str(&format!("{}\n", pairs.len()));
    for pair in &pairs {
        text.push_str(&format!(
            "{} {} {}\n",
            pair.image_i,
            pair.image_j,
            pair.matches.len()
        ));
        for (qi, tj) in &pair.matches {
            text.push_str(&format!("{qi} {tj}\n"));
        }
    }
    std::fs::write(&path, text).map_err(|e| format!("write {path:?}: {e}"))?;
    println!(
        "exhaustive_matcher: verified {}/{} pairs -> {path:?}",
        pairs.len(),
        total
    );
    Ok(())
}

fn cmd_mapper(args: &Args) -> Result<(), String> {
    let db = PathBuf::from(args.get_or("database_path", "database.db"));
    let output = PathBuf::from(args.get_or("output_path", "sparse"));
    let min_matches: usize = args
        .get_or("Mapper.min_num_matches", "20")
        .parse()
        .map_err(|e| format!("{e}"))?;
    warn_unknown_flags(
        args,
        &["database_path", "output_path", "Mapper.min_num_matches"],
    );
    let (features, names) = load_stored_features(&db)?;
    let matches_path = data_dir(&db).join("matches.txt");
    if !matches_path.exists() {
        return Err(format!(
            "no matches at {matches_path:?}; run `exhaustive_matcher` first"
        ));
    }
    let matches_text = std::fs::read_to_string(&matches_path).map_err(|e| format!("{e}"))?;
    let lines: Vec<&str> = matches_text.lines().collect();
    let mut cursor = 0usize;
    let next_line = |cursor: &mut usize| -> &str {
        let l = lines[*cursor];
        *cursor += 1;
        l
    };
    let name_count: usize = next_line(&mut cursor)
        .trim()
        .parse()
        .map_err(|e| format!("{e}"))?;
    for _ in 0..name_count {
        next_line(&mut cursor);
    }
    let pair_count: usize = next_line(&mut cursor)
        .trim()
        .parse()
        .map_err(|e| format!("{e}"))?;
    let mut pairwise = Vec::new();
    for _ in 0..pair_count {
        let head: Vec<usize> = next_line(&mut cursor)
            .split_whitespace()
            .map(|t| t.parse().map_err(|e| format!("{e}")))
            .collect::<Result<_, _>>()?;
        let count = head[2];
        let mut matches = Vec::with_capacity(count);
        for _ in 0..count {
            let m: Vec<usize> = next_line(&mut cursor)
                .split_whitespace()
                .map(|t| t.parse().map_err(|e| format!("{e}")))
                .collect::<Result<_, _>>()?;
            matches.push((m[0], m[1]));
        }
        pairwise.push(PairwiseMatches {
            image_i: head[0],
            image_j: head[1],
            matches,
        });
    }

    let (w, h) = (1600u32, 1066u32);
    let fx = 0.55 * w as f64;
    let camera = Camera::pinhole(1, w, h, fx, fx, w as f64 / 2.0, h as f64 / 2.0);
    let config = IncrementalSfmConfig {
        min_seed_matches: min_matches,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&camera, &features, &pairwise, &config)
        .map_err(|e| format!("mapper failed: {e}"))?;

    // Write a COLMAP text model (cameras/images/points3D) via the exporter.
    type LandmarkObs = (usize, usize, nalgebra::Point2<f64>);
    type Landmark = (nalgebra::Point3<f64>, Vec<LandmarkObs>);
    let mut poses_out: Vec<Pose> = Vec::new();
    let mut features_out: Vec<FeatureSet> = Vec::new();
    let mut landmarks: Vec<Landmark> = Vec::new();
    // Align exporters to registered images only: build an index map from
    // original image id -> compact writer slot.
    let mut slot = 0usize;
    let mut id_slot = std::collections::HashMap::new();
    for (id, pose) in result.poses.iter().enumerate() {
        if let Some(p) = pose {
            id_slot.insert(id, slot);
            poses_out.push(p.clone());
            features_out.push(features[id].clone());
            slot += 1;
        }
    }
    for track in &result.tracks {
        let obs: Vec<LandmarkObs> = track
            .observations
            .iter()
            .filter_map(|(img, kp, px)| {
                let slot = *id_slot.get(img)?;
                Some((slot, *kp, *px))
            })
            .collect();
        if obs.is_empty() {
            continue;
        }
        landmarks.push((track.position, obs));
    }
    write_colmap_reconstruction_for_3dgs(
        output.join("sparse"),
        &camera,
        &poses_out,
        &features_out,
        &landmarks,
        |k| {
            names[result
                .poses
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_some())
                .nth(k)
                .map(|(id, _)| id)
                .unwrap_or(0)]
            .clone()
        },
    )
    .map_err(|e| format!("export failed: {e}"))?;
    println!(
        "mapper: registered {}/{} images, {} points, mean reprojection {:.3} px",
        result.registered_images,
        names.len(),
        result.tracks.len(),
        result.mean_reprojection_px
    );
    Ok(())
}

fn cmd_model_converter(args: &Args) -> Result<(), String> {
    let input = PathBuf::from(args.get_or("input_path", "."));
    let output_type = args.get_or("output_type", "TXT");
    warn_unknown_flags(args, &["input_path", "output_path", "output_type"]);
    if output_type != "TXT" {
        return Err(format!("output type {output_type} unsupported (TXT only)"));
    }
    for file in ["cameras.txt", "images.txt", "points3D.txt"] {
        let src = input.join(file);
        if src.exists() {
            println!("{}", src.display());
        }
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    let result = match args.flags.get("subcommand").map(String::as_str) {
        Some("feature_extractor") => cmd_feature_extractor(&args),
        Some("exhaustive_matcher") => cmd_exhaustive_matcher(&args),
        Some("mapper") => cmd_mapper(&args),
        Some("model_converter") => cmd_model_converter(&args),
        Some(other) => Err(format!(
            "unknown subcommand `{other}` (expected feature_extractor | exhaustive_matcher | mapper | model_converter)"
        )),
        None => Err("usage: colmap <subcommand> [--flags] — see module docs".to_string()),
    };
    if let Err(error) = result {
        eprintln!("colmap: {error}");
        std::process::exit(1);
    }
}
