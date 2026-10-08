//! Feature extraction, export, canonical ordering and imported feature / match files.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum MapperKind {
    Incremental,
    Global,
    /// Incremental first, then global with those poses as absolute priors.
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum FeatureExtractorKind {
    Files,
    Sift,
}

/// Per-row SIFT metadata retained only for the opt-in orientation-locus
/// canonicalizer.  `FeatureSet` deliberately remains the public, compact
/// `(x,y)+descriptor` representation used by every mapper path.  A detector
/// extremum that emitted several orientation rows has identical `(x,y,scale)`
/// metadata; rows with different scales remain different loci.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FeatureLocusMetadata {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) scale: f64,
    pub(super) orientation: f64,
}

/// Quantized physical identity of one detector locus.  The key is derived
/// from metadata rather than the source row index, so it survives feature
/// permutations and feature-file round trips.  Orientation is intentionally
/// absent: orientation copies are alternatives of one locus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct FeatureLocusKey {
    x: i64,
    y: i64,
    scale: i64,
}

const FEATURE_LOCUS_COORD_SCALE: f64 = 1_000_000.0;

pub(super) fn feature_locus_key(metadata: FeatureLocusMetadata) -> Option<FeatureLocusKey> {
    let quantize = |value: f64| {
        let scaled = value * FEATURE_LOCUS_COORD_SCALE;
        (scaled.is_finite() && scaled >= i64::MIN as f64 && scaled <= i64::MAX as f64)
            .then(|| scaled.round() as i64)
    };
    Some(FeatureLocusKey {
        x: quantize(metadata.x)?,
        y: quantize(metadata.y)?,
        scale: quantize(metadata.scale.abs())?,
    })
}

/// Extract SIFT features for one image path.
///
/// When `extra_keypoints > 0`, extracts a primary set at `max_keypoints` then
/// appends spatially novel survivors from a denser extraction so the primary
/// contrast-ranked prefix stays identical to a plain `--sift-max-keypoints`
/// run (load-bearing for courtyard hub edge `10-11`).
#[cfg(feature = "image-io")]
fn extract_sift_for_image(
    path: &Path,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
    expected_dimensions: Option<(u32, u32)>,
) -> Result<
    (
        FeatureSet,
        usize,
        Option<Vec<Vec<f32>>>,
        Vec<FeatureLocusMetadata>,
    ),
    Box<dyn std::error::Error>,
> {
    use visloc_rs::vision::features::sift::{
        describe_sift_keypoints, GrayImage, SiftConfig, SiftDetector, SiftNormalization,
    };
    let detector_grayscale = if colmap_compatible_grayscale || split_colmap_detector_grayscale {
        visloc_io::images::read_common_image_colmap_grayscale(path)?
    } else {
        visloc_io::images::read_common_image(path)?
    };
    if let Some((expected_width, expected_height)) = expected_dimensions {
        let actual = (
            detector_grayscale.width() as u32,
            detector_grayscale.height() as u32,
        );
        if actual != (expected_width, expected_height) {
            return Err(format!(
                "source image {path:?} dimensions {}x{} do not match calibration {}x{}",
                actual.0, actual.1, expected_width, expected_height
            )
            .into());
        }
    }
    let descriptor_grayscale = if split_colmap_detector_grayscale {
        Some(visloc_io::images::read_common_image(path)?)
    } else {
        None
    };
    let image = GrayImage::new(
        detector_grayscale.width(),
        detector_grayscale.height(),
        detector_grayscale.pixels(),
    )?;
    let descriptor_image = if let Some(grayscale) = descriptor_grayscale.as_ref() {
        GrayImage::new(grayscale.width(), grayscale.height(), grayscale.pixels())?
    } else {
        GrayImage::new(image.width, image.height, image.pixels)?
    };
    let detector = match detector {
        "dog" => SiftDetector::Dog,
        "hessian-laplace" | "hessian" => SiftDetector::HessianLaplace,
        other => {
            return Err(format!("unknown --sift-detector {other} (dog|hessian-laplace)").into())
        }
    };
    let make_cfg = |cap: usize, threshold: f64| SiftConfig {
        max_keypoints: cap,
        affine,
        detector,
        multi_anisotropy: multi_anisotropy && affine,
        domain_size_pooling: dsp,
        dsp_num_scales: if dsp { dsp_num_scales.max(1) } else { 15 },
        normalization: if l1_root {
            SiftNormalization::L1Root
        } else {
            SiftNormalization::L2
        },
        max_orientations,
        standard_orientation_peaks: standard_orientations,
        prefer_larger_scale,
        full_pyramid,
        descriptor_magnification,
        scale_adaptive_gradients,
        vlfeat_compatible_descriptor,
        vlfeat_compatible_detector,
        vlfeat_bilinear_orientations,
        vlfeat_compatible_output_order,
        contrast_threshold: threshold,
        ..SiftConfig::default()
    };
    let primary_config = make_cfg(max_keypoints, contrast_threshold);
    let (mut keypoints, mut descriptors) = if split_colmap_detector_grayscale {
        extract_sift_with_split_grayscale(&image, &descriptor_image, &primary_config)?
    } else {
        extract_sift_maybe_gpu(&image, &primary_config)?
    };
    let primary_keypoint_count = keypoints.len();
    if extra_keypoints > 0 {
        let dense_cap =
            max_keypoints.saturating_add(extra_keypoints.saturating_mul(2).max(extra_keypoints));
        let dense_threshold =
            effective_extra_contrast_threshold(extra_contrast_threshold, contrast_threshold);
        let (dense_kp, dense_desc) =
            extract_sift_maybe_gpu(&image, &make_cfg(dense_cap, dense_threshold))?;
        append_spatially_novel_keypoints(
            &mut keypoints,
            &mut descriptors,
            dense_kp,
            dense_desc,
            extra_keypoints,
        );
        if split_colmap_detector_grayscale {
            descriptors = describe_sift_keypoints(
                &descriptor_image,
                &keypoints,
                &make_cfg(keypoints.len(), contrast_threshold),
            );
        }
    }
    let alternate_descriptors = append_descriptor_magnification.map(|magnification| {
        let alternate_config = SiftConfig {
            descriptor_magnification: magnification,
            max_keypoints: keypoints.len(),
            ..make_cfg(keypoints.len(), contrast_threshold)
        };
        let descriptor_source = if split_colmap_detector_grayscale {
            &descriptor_image
        } else {
            &image
        };
        let descriptors = describe_sift_keypoints(descriptor_source, &keypoints, &alternate_config);
        assert_eq!(
            descriptors.len(),
            keypoints.len(),
            "alternate descriptor bank must preserve keypoint indices"
        );
        descriptors
    });
    let features = FeatureSet::new(
        keypoints.iter().map(|k| Point2::new(k.x, k.y)).collect(),
        descriptors,
    )?;
    let locus_metadata = keypoints
        .iter()
        .map(|keypoint| FeatureLocusMetadata {
            x: keypoint.x,
            y: keypoint.y,
            scale: keypoint.sigma,
            orientation: keypoint.orientation,
        })
        .collect();
    Ok((
        features,
        primary_keypoint_count,
        alternate_descriptors,
        locus_metadata,
    ))
}

/// Detect keypoints on one grayscale image and describe those exact
/// keypoints on another.  The split is intentionally narrow: it is used only
/// by the opt-in COLMAP preprocessing experiment, and never redetects or
/// changes keypoint order while switching the descriptor source.
#[cfg(feature = "image-io")]
pub(super) fn extract_sift_with_split_grayscale(
    detector_image: &GrayImage<'_>,
    descriptor_image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftError> {
    let (keypoints, _) = extract_sift(detector_image, config)?;
    let descriptors = describe_sift_keypoints(descriptor_image, &keypoints, config);
    Ok((keypoints, descriptors))
}

/// Resolve the optional extra-extraction threshold without changing the
/// legacy path: absent means exactly the primary SIFT threshold.
#[cfg(feature = "image-io")]
pub(super) fn effective_extra_contrast_threshold(extra: Option<f64>, primary: f64) -> f64 {
    extra.unwrap_or(primary)
}

/// Append at most `max_extra` dense SIFT detections that are novel on the
/// existing 0.5 px spatial grid. The primary vectors are mutated only by
/// appending, so their prefix remains byte-identical to the primary extraction.
#[cfg(feature = "image-io")]
pub(super) fn append_spatially_novel_keypoints(
    keypoints: &mut Vec<visloc_rs::vision::features::sift::SiftKeypoint>,
    descriptors: &mut Vec<Vec<f32>>,
    dense_keypoints: Vec<visloc_rs::vision::features::sift::SiftKeypoint>,
    dense_descriptors: Vec<Vec<f32>>,
    max_extra: usize,
) -> usize {
    let mut seen: HashSet<(i32, i32)> = keypoints
        .iter()
        .map(|k| ((k.x * 2.0).round() as i32, (k.y * 2.0).round() as i32))
        .collect();
    let mut added = 0usize;
    for (keypoint, descriptor) in dense_keypoints
        .into_iter()
        .zip(dense_descriptors.into_iter())
    {
        if added >= max_extra {
            break;
        }
        let key = (
            (keypoint.x * 2.0).round() as i32,
            (keypoint.y * 2.0).round() as i32,
        );
        if seen.insert(key) {
            keypoints.push(keypoint);
            descriptors.push(descriptor);
            added += 1;
        }
    }
    added
}

/// Enumerate supported SIFT source images in the same lexical order used by
/// the historical batch extractor.  Keeping this as a shared helper makes the
/// streaming export byte-comparable with the ordinary path.
#[cfg(feature = "image-io")]
pub(super) fn list_sift_image_paths(
    dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
                    )
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// In-process SIFT over every common-format image in `dir`, sorted lexically.
#[cfg(not(feature = "image-io"))]
pub(super) fn load_images_with_sift(
    _dir: &Path,
    _max_keypoints: usize,
    _affine: bool,
    _detector: &str,
    _multi_anisotropy: bool,
    _dsp: bool,
    _dsp_num_scales: usize,
    _l1_root: bool,
    _max_orientations: usize,
    _standard_orientations: bool,
    _prefer_larger_scale: bool,
    _full_pyramid: bool,
    _contrast_threshold: f64,
    _descriptor_magnification: f64,
    _scale_adaptive_gradients: bool,
    _vlfeat_compatible_descriptor: bool,
    _vlfeat_compatible_detector: bool,
    _vlfeat_bilinear_orientations: bool,
    _vlfeat_compatible_output_order: bool,
    _colmap_compatible_grayscale: bool,
    _split_colmap_detector_grayscale: bool,
    _append_descriptor_magnification: Option<f64>,
    _extra_stems: &[String],
    _extra_keypoints: usize,
    _extra_contrast_threshold: Option<f64>,
) -> Result<
    (
        Vec<FeatureSet>,
        Vec<String>,
        Vec<usize>,
        Vec<Option<Vec<Vec<f32>>>>,
        Vec<Option<Vec<FeatureLocusMetadata>>>,
    ),
    Box<dyn std::error::Error>,
> {
    Err("--feature-extractor sift requires building with --features image-io".into())
}

/// In-process SIFT over every common-format image in `dir`, sorted lexically.
#[cfg(feature = "image-io")]
pub(super) fn load_images_with_sift(
    dir: &Path,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_stems: &[String],
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
) -> Result<
    (
        Vec<FeatureSet>,
        Vec<String>,
        Vec<usize>,
        Vec<Option<Vec<Vec<f32>>>>,
        Vec<Option<Vec<FeatureLocusMetadata>>>,
    ),
    Box<dyn std::error::Error>,
> {
    let paths = list_sift_image_paths(dir)?;
    let total = paths.len();
    let extra_want: HashSet<&str> = extra_stems.iter().map(String::as_str).collect();
    eprintln!(
        "sift: extracting {total} image(s) (dsp={dsp}, dsp_scales={}, l1_root={l1_root}, contrast={contrast_threshold}, vlfeat_detector={vlfeat_compatible_detector}, colmap_gray={colmap_compatible_grayscale}, split_colmap_gray={split_colmap_detector_grayscale}, extra_kp={extra_keypoints} stems={extra_stems:?})",
        if dsp { dsp_num_scales } else { 0 }
    );
    let results: Result<Vec<_>, Box<dyn std::error::Error + Send>> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            let started = std::time::Instant::now();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let extra = if extra_keypoints > 0 && extra_want.contains(stem) {
                extra_keypoints
            } else {
                0
            };
            let (feat, primary_keypoint_count, alternate_descriptors, locus_metadata) =
                extract_sift_for_image(
                    path,
                    max_keypoints,
                    affine,
                    detector,
                    multi_anisotropy,
                    dsp,
                    dsp_num_scales,
                    l1_root,
                    max_orientations,
                    standard_orientations,
                    prefer_larger_scale,
                    full_pyramid,
                    contrast_threshold,
                    descriptor_magnification,
                    scale_adaptive_gradients,
                    vlfeat_compatible_descriptor,
                    vlfeat_compatible_detector,
                    vlfeat_bilinear_orientations,
                    vlfeat_compatible_output_order,
                    colmap_compatible_grayscale,
                    split_colmap_detector_grayscale,
                    append_descriptor_magnification,
                    extra,
                    extra_contrast_threshold,
                    None,
                )
                .map_err(|e| -> Box<dyn std::error::Error + Send> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image")
                .to_string();
            eprintln!(
                "sift: [{}/{}] {} -> {} kp (primary {}) in {:.1}s",
                idx + 1,
                total,
                name,
                feat.keypoints.len(),
                primary_keypoint_count,
                started.elapsed().as_secs_f64()
            );
            Ok((
                feat,
                name,
                primary_keypoint_count,
                alternate_descriptors,
                locus_metadata,
            ))
        })
        .collect();
    let results = results.map_err(|e| -> Box<dyn std::error::Error> { e })?;
    let mut features = Vec::with_capacity(results.len());
    let mut names = Vec::with_capacity(results.len());
    let mut primary_keypoint_counts = Vec::with_capacity(results.len());
    let mut alternate_descriptors = Vec::with_capacity(results.len());
    let mut locus_metadata = Vec::with_capacity(results.len());
    for (feature, name, primary_keypoint_count, alternate, loci) in results {
        features.push(feature);
        names.push(name);
        primary_keypoint_counts.push(primary_keypoint_count);
        alternate_descriptors.push(alternate);
        locus_metadata.push(Some(loci));
    }
    Ok((
        features,
        names,
        primary_keypoint_counts,
        alternate_descriptors,
        locus_metadata,
    ))
}

/// Extract and export SIFT one image at a time.  This intentionally shares
/// the same per-image extractor and serializers as the batch path, but avoids
/// Rayon and does not retain completed source images or descriptor banks.
#[cfg(feature = "image-io")]
pub(super) fn stream_export_images_with_sift(
    dir: &Path,
    output_dir: &Path,
    calibration_model_dir: Option<&Path>,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_stems: &[String],
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
    resume: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    let paths = list_sift_image_paths(dir)?;
    let image_names = paths
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("source image {path:?} has no UTF-8 filename"))
                .map(str::to_owned)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let calibration = calibration_model_dir
        .map(|model_dir| resolve_input_colmap_calibration(model_dir, &image_names))
        .transpose()
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    let extra_want: HashSet<&str> = extra_stems.iter().map(String::as_str).collect();
    eprintln!(
        "sift-stream: extracting {} image(s) (output={output_dir:?}, calibration={}, resume={resume}, dsp={dsp}, contrast={contrast_threshold}, vlfeat_detector={vlfeat_compatible_detector})",
        image_names.len(),
        calibration.is_some(),
    );
    let mut stems = HashSet::new();
    for name in &image_names {
        let stem = image_stem(name);
        if !stems.insert(stem.to_owned()) {
            return Err(format!("duplicate source image stem {stem:?}").into());
        }
    }
    std::fs::create_dir_all(output_dir)?;
    let mut total_keypoints = 0usize;
    for (index, path) in paths.iter().enumerate() {
        let stem = image_stem(&image_names[index]);
        let extra = if extra_keypoints > 0 && extra_want.contains(stem) {
            extra_keypoints
        } else {
            0
        };
        let expected_dimensions = calibration.as_ref().map(|loaded| {
            let camera = &loaded.native_cameras[index];
            (camera.width, camera.height)
        });
        let expected_camera = calibration
            .as_ref()
            .map(|loaded| &loaded.native_cameras[index]);
        let config_hash = sift_stream_config_hash(
            &image_names[index],
            max_keypoints,
            affine,
            detector,
            multi_anisotropy,
            dsp,
            dsp_num_scales,
            l1_root,
            max_orientations,
            standard_orientations,
            prefer_larger_scale,
            full_pyramid,
            contrast_threshold,
            descriptor_magnification,
            scale_adaptive_gradients,
            vlfeat_compatible_descriptor,
            vlfeat_compatible_detector,
            vlfeat_bilinear_orientations,
            vlfeat_compatible_output_order,
            colmap_compatible_grayscale,
            split_colmap_detector_grayscale,
            append_descriptor_magnification,
            extra,
            extra_contrast_threshold,
            expected_dimensions,
            expected_camera,
        );
        let feature_path = output_dir.join(format!("{stem}_features.txt"));
        let metadata_path = output_dir.join(format!("{stem}_loci.txt"));
        let manifest_path = sift_stream_manifest_path(output_dir, stem);
        let source_digest = if resume {
            Some(file_fnv1a64(path)?)
        } else {
            None
        };
        if let Some(source_digest) = source_digest {
            if let Some(rows) = validate_sift_stream_manifest(
                &manifest_path,
                config_hash,
                source_digest,
                &feature_path,
                &metadata_path,
            )? {
                total_keypoints = total_keypoints.saturating_add(rows);
                eprintln!(
                    "sift-stream: [{}/{}] {} -> {} kp (resumed)",
                    index + 1,
                    image_names.len(),
                    image_names[index],
                    rows,
                );
                continue;
            }
            eprintln!(
                "sift-stream: [{}/{}] {} has no valid completion sidecar; re-extracting",
                index + 1,
                image_names.len(),
                image_names[index],
            );
        }
        let started = std::time::Instant::now();
        let (features, _primary_count, _alternate, loci) = extract_sift_for_image(
            path,
            max_keypoints,
            affine,
            detector,
            multi_anisotropy,
            dsp,
            dsp_num_scales,
            l1_root,
            max_orientations,
            standard_orientations,
            prefer_larger_scale,
            full_pyramid,
            contrast_threshold,
            descriptor_magnification,
            scale_adaptive_gradients,
            vlfeat_compatible_descriptor,
            vlfeat_compatible_detector,
            vlfeat_bilinear_orientations,
            vlfeat_compatible_output_order,
            colmap_compatible_grayscale,
            split_colmap_detector_grayscale,
            append_descriptor_magnification,
            extra,
            extra_contrast_threshold,
            expected_dimensions,
        )?;
        if features.keypoints.len() != loci.len() {
            return Err(format!(
                "SIFT extractor returned {} keypoints but {} locus rows for {path:?}",
                features.keypoints.len(),
                loci.len()
            )
            .into());
        }
        write_stream_file_atomically(&feature_path, &feature_export_text(&features))?;
        write_stream_file_atomically(&metadata_path, &locus_metadata_text(&loci))?;
        if let Some(source_digest) = source_digest {
            let feature_digest = file_fnv1a64(&feature_path)?;
            let metadata_digest = file_fnv1a64(&metadata_path)?;
            write_sift_stream_manifest_atomically(
                &manifest_path,
                config_hash,
                source_digest,
                features.len(),
                feature_digest,
                metadata_digest,
            )?;
        }
        total_keypoints = total_keypoints.saturating_add(features.len());
        eprintln!(
            "sift-stream: [{}/{}] {} -> {} kp in {:.1}s",
            index + 1,
            image_names.len(),
            image_names[index],
            features.len(),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(total_keypoints)
}

pub(super) fn image_name_for(feat_filename: &str, feat_suffix: &str, image_suffix: &str) -> String {
    match feat_filename.strip_suffix(feat_suffix) {
        Some(stem) => format!("{stem}{image_suffix}"),
        None => feat_filename.to_string(),
    }
}

pub(super) fn export_features_to_dir(
    dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    locus_metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(), Box<dyn std::error::Error>> {
    export_features_to_dir_impl(dir, image_names, features, None, locus_metadata)
}

/// Export canonical descriptors with a parallel native-pixel keypoint sidecar.
/// This is the calibration-aware counterpart to [`export_features_to_dir`];
/// descriptors remain owned by the mapper feature bank and are never cloned
/// merely to restore native coordinates for export.
pub(super) fn export_features_to_dir_with_native_keypoints(
    dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    native_keypoints: &[Vec<Point2<f64>>],
    locus_metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(), Box<dyn std::error::Error>> {
    export_features_to_dir_impl(
        dir,
        image_names,
        features,
        Some(native_keypoints),
        locus_metadata,
    )
}

fn export_features_to_dir_impl(
    dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    native_keypoints: Option<&[Vec<Point2<f64>>]>,
    locus_metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(native_keypoints) = native_keypoints {
        if native_keypoints.len() != features.len() {
            return Err(format!(
                "native feature export: {} keypoint sets but {} feature sets",
                native_keypoints.len(),
                features.len()
            )
            .into());
        }
        for (image, (feature_set, keypoints)) in features.iter().zip(native_keypoints).enumerate() {
            if feature_set.descriptors.len() != keypoints.len() {
                return Err(format!(
                    "native feature export: image {image} has {} descriptors but {} keypoints",
                    feature_set.descriptors.len(),
                    keypoints.len()
                )
                .into());
            }
        }
    }
    std::fs::create_dir_all(dir)?;
    for (image_index, (name, feat)) in image_names.iter().zip(features).enumerate() {
        let stem = Path::new(name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name);
        let path = dir.join(format!("{stem}_features.txt"));
        let keypoints =
            native_keypoints.map_or(feat.keypoints.as_slice(), |all| all[image_index].as_slice());
        std::fs::write(
            &path,
            feature_export_text_with_keypoints(keypoints, &feat.descriptors),
        )?;
        if let Some(loci) = locus_metadata
            .get(image_index)
            .and_then(|loci| loci.as_ref())
        {
            let metadata_path = dir.join(format!("{stem}_loci.txt"));
            std::fs::write(metadata_path, locus_metadata_text(loci))?;
        }
    }
    Ok(())
}

#[cfg_attr(not(feature = "image-io"), allow(dead_code))]
pub(super) fn feature_export_text(features: &FeatureSet) -> String {
    feature_export_text_with_keypoints(&features.keypoints, &features.descriptors)
}

pub(super) fn feature_export_text_with_keypoints(
    keypoints: &[Point2<f64>],
    descriptors: &[Vec<f32>],
) -> String {
    let mut out = String::from("# visloc external-deep feature export\n");
    for (kp, desc) in keypoints.iter().zip(descriptors.iter()) {
        out.push_str(&format!("{:.6} {:.6} {:.6}", kp.x, kp.y, 1.0));
        for value in desc.as_slice() {
            out.push_str(&format!(" {value:.6}"));
        }
        out.push('\n');
    }
    out
}

pub(super) fn locus_metadata_text(loci: &[FeatureLocusMetadata]) -> String {
    let mut out = String::from("# visloc orientation-locus metadata: x y scale orientation\n");
    for locus in loci {
        out.push_str(&format!(
            "{:.17e} {:.17e} {:.17e} {:.17e}\n",
            locus.x, locus.y, locus.scale, locus.orientation
        ));
    }
    out
}

/// Write one completed feature result with a same-directory temporary file
/// followed by an atomic rename.  A failed extractor therefore cannot leave a
/// truncated final feature file in the requested export directory.
#[cfg(feature = "image-io")]
fn write_stream_file_atomically(
    path: &Path,
    contents: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid feature export path {path:?}"))?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    std::fs::write(&temporary, contents)?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

/// Per-image completion marker for resumable SIFT stream exports.  The
/// marker is deliberately separate from the feature/locus files: it is
/// atomically installed last, so its presence never by itself makes a
/// partially written pair look complete.
#[cfg(feature = "image-io")]
const SIFT_STREAM_MANIFEST_MAGIC: &str = "visloc_sift_stream_manifest_v1";

#[cfg(feature = "image-io")]
pub(super) fn sift_stream_manifest_path(output_dir: &Path, stem: &str) -> PathBuf {
    output_dir.join(format!("{stem}_sift_stream_manifest.txt"))
}

/// Return `(byte_count, FNV-1a-64)` for one completed source or output file.
/// This is a stable corruption/configuration guard, not a cryptographic
/// authenticity claim; the benchmark's top-level manifests use SHA-256 when
/// a stronger artifact identity is required.
#[cfg(feature = "image-io")]
pub(super) fn file_fnv1a64(path: &Path) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; 1024 * 1024];
    let mut bytes = 0u64;
    let mut hash = 0xcbf29ce484222325u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| format!("file {path:?} is too large to hash"))?;
        hash = fnv1a64_bytes_with_seed(hash, &buffer[..read]);
    }
    Ok((bytes, hash))
}

#[cfg(feature = "image-io")]
fn fnv1a64_bytes_with_seed(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

#[cfg(feature = "image-io")]
#[allow(clippy::too_many_arguments)]
fn sift_stream_config_hash(
    image_name: &str,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
    expected_dimensions: Option<(u32, u32)>,
    expected_camera: Option<&Camera>,
) -> u64 {
    // Keep this explicit rather than hashing `Args`: `--sift-stream-resume`
    // itself must not make a previously completed export stale, while every
    // extraction-affecting option and per-image camera assignment must.
    let snapshot = format!(
        "visloc_sift_stream_resume_v1;image_name={image_name:?};max_keypoints={max_keypoints};affine={affine};detector={detector:?};multi_anisotropy={multi_anisotropy};dsp={dsp};dsp_num_scales={dsp_num_scales};l1_root={l1_root};max_orientations={max_orientations};standard_orientations={standard_orientations};prefer_larger_scale={prefer_larger_scale};full_pyramid={full_pyramid};contrast_threshold={contrast_threshold:?};descriptor_magnification={descriptor_magnification:?};scale_adaptive_gradients={scale_adaptive_gradients};vlfeat_compatible_descriptor={vlfeat_compatible_descriptor};vlfeat_compatible_detector={vlfeat_compatible_detector};vlfeat_bilinear_orientations={vlfeat_bilinear_orientations};vlfeat_compatible_output_order={vlfeat_compatible_output_order};colmap_compatible_grayscale={colmap_compatible_grayscale};split_colmap_detector_grayscale={split_colmap_detector_grayscale};append_descriptor_magnification={append_descriptor_magnification:?};extra_keypoints={extra_keypoints};extra_contrast_threshold={extra_contrast_threshold:?};expected_dimensions={expected_dimensions:?};expected_camera={expected_camera:?}"
    );
    effective_config_hash(&snapshot)
}

#[cfg(feature = "image-io")]
fn parse_sift_stream_u64(fields: &HashMap<String, String>, key: &str) -> Option<u64> {
    fields.get(key)?.parse().ok()
}

#[cfg(feature = "image-io")]
fn parse_sift_stream_hex(fields: &HashMap<String, String>, key: &str) -> Option<u64> {
    u64::from_str_radix(fields.get(key)?, 16).ok()
}

#[cfg(feature = "image-io")]
fn count_sift_stream_rows(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut rows = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            rows = rows
                .checked_add(1)
                .ok_or_else(|| format!("row count overflow in {path:?}"))?;
        }
    }
    Ok(rows)
}

#[cfg(feature = "image-io")]
pub(super) fn write_sift_stream_manifest_atomically(
    path: &Path,
    config_hash: u64,
    source_digest: (u64, u64),
    feature_rows: usize,
    feature_digest: (u64, u64),
    loci_digest: (u64, u64),
) -> Result<(), Box<dyn std::error::Error>> {
    let contents = format!(
        "{SIFT_STREAM_MANIFEST_MAGIC}\nconfig_fnv1a64={config_hash:016x}\nsource_bytes={}\nsource_fnv1a64={:016x}\nfeature_rows={feature_rows}\nfeature_bytes={}\nfeature_fnv1a64={:016x}\nloci_bytes={}\nloci_fnv1a64={:016x}\n",
        source_digest.0,
        source_digest.1,
        feature_digest.0,
        feature_digest.1,
        loci_digest.0,
        loci_digest.1,
    );
    write_stream_file_atomically(path, &contents)
}

/// Validate a completion marker and all files it covers.  `Ok(None)` means
/// that the output is absent, malformed, stale, or tampered and must be
/// re-extracted; filesystem errors unrelated to absence remain hard errors.
#[cfg(feature = "image-io")]
pub(super) fn validate_sift_stream_manifest(
    path: &Path,
    expected_config_hash: u64,
    expected_source_digest: (u64, u64),
    feature_path: &Path,
    loci_path: &Path,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut lines = text.lines();
    if lines.next() != Some(SIFT_STREAM_MANIFEST_MAGIC) {
        return Ok(None);
    }
    let mut fields = HashMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Ok(None);
        };
        if key.is_empty()
            || value.is_empty()
            || fields.insert(key.to_owned(), value.to_owned()).is_some()
        {
            return Ok(None);
        }
    }
    if fields.len() != 8 {
        return Ok(None);
    }
    let config_hash = parse_sift_stream_hex(&fields, "config_fnv1a64");
    let source_bytes = parse_sift_stream_u64(&fields, "source_bytes");
    let source_hash = parse_sift_stream_hex(&fields, "source_fnv1a64");
    let rows = fields
        .get("feature_rows")
        .and_then(|value| value.parse().ok());
    let feature_bytes = parse_sift_stream_u64(&fields, "feature_bytes");
    let feature_hash = parse_sift_stream_hex(&fields, "feature_fnv1a64");
    let loci_bytes = parse_sift_stream_u64(&fields, "loci_bytes");
    let loci_hash = parse_sift_stream_hex(&fields, "loci_fnv1a64");
    let (
        Some(config_hash),
        Some(source_bytes),
        Some(source_hash),
        Some(rows),
        Some(feature_bytes),
        Some(feature_hash),
        Some(loci_bytes),
        Some(loci_hash),
    ) = (
        config_hash,
        source_bytes,
        source_hash,
        rows,
        feature_bytes,
        feature_hash,
        loci_bytes,
        loci_hash,
    )
    else {
        return Ok(None);
    };
    if config_hash != expected_config_hash || (source_bytes, source_hash) != expected_source_digest
    {
        return Ok(None);
    }
    let actual_feature = match file_fnv1a64(feature_path) {
        Ok(digest) => digest,
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    let actual_loci = match file_fnv1a64(loci_path) {
        Ok(digest) => digest,
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    if actual_feature != (feature_bytes, feature_hash) || actual_loci != (loci_bytes, loci_hash) {
        return Ok(None);
    }
    if count_sift_stream_rows(feature_path)? != rows || count_sift_stream_rows(loci_path)? != rows {
        return Ok(None);
    }
    Ok(Some(rows))
}

/// Streaming counterpart to [`export_features_to_dir`].  The loader callback
/// is invoked exactly once per path and its result is consumed and dropped
/// before the next callback, so source images, pyramids, and descriptor banks
/// are never accumulated across the directory.
#[cfg(feature = "image-io")]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn stream_export_features_with_loader<F>(
    paths: &[PathBuf],
    output_dir: &Path,
    mut loader: F,
) -> Result<usize, Box<dyn std::error::Error>>
where
    F: FnMut(
        usize,
        &Path,
    ) -> Result<(FeatureSet, Vec<FeatureLocusMetadata>), Box<dyn std::error::Error>>,
{
    let mut stems = HashSet::new();
    let mut stem_names = Vec::with_capacity(paths.len());
    for path in paths {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("source image {path:?} has no UTF-8 stem"))?
            .to_owned();
        if !stems.insert(stem.clone()) {
            return Err(format!("duplicate source image stem {stem:?}").into());
        }
        stem_names.push(stem);
    }
    std::fs::create_dir_all(output_dir)?;
    let mut total_keypoints = 0usize;
    for (index, path) in paths.iter().enumerate() {
        let stem = &stem_names[index];
        let (features, loci) = loader(index, path)?;
        if features.keypoints.len() != loci.len() {
            return Err(format!(
                "stream loader returned {} keypoints but {} locus rows for {path:?}",
                features.keypoints.len(),
                loci.len()
            )
            .into());
        }
        let feature_path = output_dir.join(format!("{stem}_features.txt"));
        let metadata_path = output_dir.join(format!("{stem}_loci.txt"));
        write_stream_file_atomically(&feature_path, &feature_export_text(&features))?;
        write_stream_file_atomically(&metadata_path, &locus_metadata_text(&loci))?;
        total_keypoints = total_keypoints.saturating_add(features.len());
        // `features` owns all descriptors returned by this callback.  It is
        // deliberately scoped to one loop iteration and dropped here before
        // the next image is decoded.
    }
    Ok(total_keypoints)
}

const CANONICAL_FEATURE_COORD_SCALE: f64 = 1_000_000.0;

fn canonical_coordinate_key(value: f64) -> (u8, i64) {
    if value.is_finite() {
        (0, (value * CANONICAL_FEATURE_COORD_SCALE).round() as i64)
    } else if value.is_nan() {
        (2, 0)
    } else if value.is_sign_negative() {
        (1, 0)
    } else {
        (3, 0)
    }
}

pub(super) fn canonical_descriptor_cmp(lhs: &[f32], rhs: &[f32]) -> CmpOrdering {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| a.total_cmp(b))
        .find(|ordering| *ordering != CmpOrdering::Equal)
        .unwrap_or_else(|| lhs.len().cmp(&rhs.len()))
}

fn canonical_feature_cmp(set: &FeatureSet, lhs: usize, rhs: usize) -> CmpOrdering {
    let lhs_point = set.keypoints[lhs];
    let rhs_point = set.keypoints[rhs];
    canonical_coordinate_key(lhs_point.x)
        .cmp(&canonical_coordinate_key(rhs_point.x))
        .then_with(|| {
            canonical_coordinate_key(lhs_point.y).cmp(&canonical_coordinate_key(rhs_point.y))
        })
        .then_with(|| canonical_descriptor_cmp(&set.descriptors[lhs], &set.descriptors[rhs]))
        .then_with(|| lhs.cmp(&rhs))
}

/// Put every feature/descriptor row into a deterministic physical order and
/// return an old-index → new-index map per image. The descriptor tie-break is
/// only used for co-located rows; exact duplicate rows remain physically
/// indistinguishable and use their original index as the final deterministic
/// fallback. Alternate SIFT banks follow the same permutation one-for-one.
pub(super) fn canonicalize_feature_order(
    features: &mut [FeatureSet],
    alternate_descriptors: &mut [Option<Vec<Vec<f32>>>],
) -> Result<Vec<Vec<usize>>, String> {
    if features.len() != alternate_descriptors.len() {
        return Err(format!(
            "canonical feature order: {} feature sets but {} alternate banks",
            features.len(),
            alternate_descriptors.len()
        ));
    }
    let mut old_to_new = Vec::with_capacity(features.len());
    for image in 0..features.len() {
        let set = &features[image];
        let mut order: Vec<usize> = (0..set.len()).collect();
        order.sort_by(|lhs, rhs| canonical_feature_cmp(set, *lhs, *rhs));
        let mut inverse = vec![0usize; order.len()];
        for (new_index, &old_index) in order.iter().enumerate() {
            inverse[old_index] = new_index;
        }
        let set = &mut features[image];
        let old_keypoints = std::mem::take(&mut set.keypoints);
        let old_descriptors = std::mem::take(&mut set.descriptors);
        set.keypoints = order.iter().map(|&index| old_keypoints[index]).collect();
        set.descriptors = order
            .iter()
            .map(|&index| old_descriptors[index].clone())
            .collect();
        if let Some(bank) = alternate_descriptors[image].as_mut() {
            if bank.len() != order.len() {
                return Err(format!(
                    "canonical feature order: image {image} alternate bank has {} rows, expected {}",
                    bank.len(),
                    order.len()
                ));
            }
            let old_bank = std::mem::take(bank);
            *bank = order.iter().map(|&index| old_bank[index].clone()).collect();
        }
        old_to_new.push(inverse);
    }
    Ok(old_to_new)
}

/// Apply the mapper's old-index → new-index permutation to the native pixel
/// sidecar used by the multi-camera exporter.  Descriptors are intentionally
/// absent from this sidecar: canonicalization changes only keypoint pixels,
/// so the mapper's descriptor bank remains the single owner of each row.
pub(super) fn remap_feature_keypoints_by_old_to_new(
    keypoints: &mut [Vec<Point2<f64>>],
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    if keypoints.len() != old_to_new.len() {
        return Err(format!(
            "native keypoint order: {} feature sets but {} index maps",
            keypoints.len(),
            old_to_new.len()
        ));
    }
    for (image, (image_keypoints, map)) in keypoints.iter_mut().zip(old_to_new).enumerate() {
        if image_keypoints.len() != map.len() || map.iter().any(|&new| new >= map.len()) {
            return Err(format!(
                "native keypoint order: image {image} has {} rows but map has {} entries",
                image_keypoints.len(),
                map.len()
            ));
        }
        let mut seen = vec![false; map.len()];
        for &new_index in map {
            if std::mem::replace(&mut seen[new_index], true) {
                return Err(format!(
                    "native keypoint order: image {image} permutation contains duplicate index {new_index}"
                ));
            }
        }
        let old_keypoints = std::mem::take(image_keypoints);
        let mut reordered = vec![Point2::new(0.0, 0.0); map.len()];
        for (old_index, &new_index) in map.iter().enumerate() {
            reordered[new_index] = old_keypoints[old_index];
        }
        *image_keypoints = reordered;
    }
    Ok(())
}

/// Replace compacted mapper feature keypoints with native-pixel coordinates
/// for COLMAP's multi-camera exporter.  The descriptor vectors are retained
/// from `output_features`; the writer consumes only the keypoint coordinates.
pub(super) fn replace_feature_keypoints_from_native(
    output_features: &mut [FeatureSet],
    source_indices: &[usize],
    native_keypoints: &[Vec<Point2<f64>>],
) -> Result<(), String> {
    if output_features.len() != source_indices.len() {
        return Err(format!(
            "native export: {} output feature sets but {} source indices",
            output_features.len(),
            source_indices.len()
        ));
    }
    for (output_index, (&source_index, output)) in source_indices
        .iter()
        .zip(output_features.iter_mut())
        .enumerate()
    {
        let source = native_keypoints.get(source_index).ok_or_else(|| {
            format!(
                "native export: source image {source_index} is outside 0..{}",
                native_keypoints.len()
            )
        })?;
        if source.len() != output.descriptors.len() {
            return Err(format!(
                "native export: output image {output_index} has {} descriptors but source image {source_index} has {} keypoints",
                output.descriptors.len(),
                source.len()
            ));
        }
        output.keypoints = source.clone();
    }
    Ok(())
}

pub(super) fn remap_locus_metadata(
    metadata: &mut [Option<Vec<FeatureLocusMetadata>>],
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    if metadata.len() != old_to_new.len() {
        return Err(format!(
            "canonical feature order: {} metadata sets but {} index maps",
            metadata.len(),
            old_to_new.len()
        ));
    }
    for (image, (loci, map)) in metadata.iter_mut().zip(old_to_new).enumerate() {
        let Some(loci) = loci.as_mut() else {
            continue;
        };
        if loci.len() != map.len() {
            return Err(format!(
                "canonical feature order: image {image} metadata has {} rows, expected {}",
                loci.len(),
                map.len()
            ));
        }
        let old_loci = std::mem::take(loci);
        let mut new_loci = vec![
            FeatureLocusMetadata {
                x: f64::NAN,
                y: f64::NAN,
                scale: f64::NAN,
                orientation: f64::NAN,
            };
            old_loci.len()
        ];
        for (old_index, &new_index) in map.iter().enumerate() {
            let Some(slot) = new_loci.get_mut(new_index) else {
                return Err(format!(
                    "canonical feature order: image {image} invalid metadata map {old_index}->{new_index}"
                ));
            };
            *slot = old_loci[old_index];
        }
        *loci = new_loci;
    }
    Ok(())
}

pub(super) fn remap_imported_matches(
    imported: &mut HashMap<(usize, usize), Vec<(usize, usize)>>,
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    for (&(image_i, image_j), matches) in imported.iter_mut() {
        for (keypoint_i, keypoint_j) in matches {
            *keypoint_i = *old_to_new
                .get(image_i)
                .and_then(|indices| indices.get(*keypoint_i))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid imported index ({image_i},{keypoint_i})"
                    )
                })?;
            *keypoint_j = *old_to_new
                .get(image_j)
                .and_then(|indices| indices.get(*keypoint_j))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid imported index ({image_j},{keypoint_j})"
                    )
                })?;
        }
    }
    Ok(())
}

pub(super) fn remap_imported_verified_pairs(
    imported: &mut [ImportedVerifiedPair],
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    for pair in imported {
        for (keypoint_i, keypoint_j) in &mut pair.matches {
            *keypoint_i = *old_to_new
                .get(pair.image_i)
                .and_then(|indices| indices.get(*keypoint_i))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid verified index ({},{})",
                        pair.image_i, *keypoint_i
                    )
                })?;
            *keypoint_j = *old_to_new
                .get(pair.image_j)
                .and_then(|indices| indices.get(*keypoint_j))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid verified index ({},{})",
                        pair.image_j, *keypoint_j
                    )
                })?;
        }
    }
    Ok(())
}

/// Parse `export_colmap_matches.py` output: image count, names, then per-pair
/// `(i, j, count)` + `count` lines of `qi tj`. Pair keys are normalized to
/// `(min(i,j), max(i,j))`.
pub(super) fn parse_imported_matches_file(
    path: &Path,
    image_names: &[String],
) -> Result<HashMap<(usize, usize), Vec<(usize, usize)>>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let name_count: usize = lines
        .next()
        .ok_or("import matches: missing image count")?
        .parse()?;
    if name_count != image_names.len() {
        return Err(format!(
            "import matches: file has {name_count} names, run has {}",
            image_names.len()
        )
        .into());
    }
    for (idx, expected_name) in image_names.iter().enumerate().take(name_count) {
        let file_name = lines
            .next()
            .ok_or("import matches: truncated image name list")?;
        if file_name != expected_name {
            return Err(format!(
                "import matches: name mismatch at {idx}: file {file_name:?} vs run {:?}",
                expected_name
            )
            .into());
        }
    }
    let pair_count: usize = lines
        .next()
        .ok_or("import matches: missing pair count")?
        .parse()?;
    let mut out = HashMap::new();
    for _ in 0..pair_count {
        let head: Vec<usize> = lines
            .next()
            .ok_or("import matches: truncated pair header")?
            .split_whitespace()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()?;
        if head.len() != 3 {
            return Err("import matches: pair header needs i j count".into());
        }
        let (mut i, mut j, count) = (head[0], head[1], head[2]);
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        let mut matches = Vec::with_capacity(count);
        for _ in 0..count {
            let m: Vec<usize> = lines
                .next()
                .ok_or("import matches: truncated correspondence")?
                .split_whitespace()
                .map(|t| t.parse())
                .collect::<Result<_, _>>()?;
            if m.len() != 2 {
                return Err("import matches: correspondence needs qi tj".into());
            }
            let (qi, tj) = if head[0] <= head[1] {
                (m[0], m[1])
            } else {
                (m[1], m[0])
            };
            matches.push((qi, tj));
        }
        out.insert((i, j), matches);
    }
    Ok(out)
}

fn parse_config_token(tok: &str) -> Result<ConfigurationType, Box<dyn std::error::Error>> {
    if let Ok(n) = tok.parse::<usize>() {
        return Ok(match n {
            0 => ConfigurationType::Undefined,
            1 => ConfigurationType::Degenerate,
            2 => ConfigurationType::Uncalibrated,
            3 => ConfigurationType::Calibrated,
            4 => ConfigurationType::Planar,
            5 => ConfigurationType::Panoramic,
            6 => ConfigurationType::PlanarOrPanoramic,
            7 => ConfigurationType::Watermark,
            8 => ConfigurationType::Multiple,
            other => return Err(format!("unknown config code {other}").into()),
        });
    }
    Ok(match tok {
        "Undefined" => ConfigurationType::Undefined,
        "Degenerate" => ConfigurationType::Degenerate,
        "Uncalibrated" => ConfigurationType::Uncalibrated,
        "Calibrated" => ConfigurationType::Calibrated,
        "Planar" => ConfigurationType::Planar,
        "Panoramic" => ConfigurationType::Panoramic,
        "PlanarOrPanoramic" => ConfigurationType::PlanarOrPanoramic,
        "Watermark" => ConfigurationType::Watermark,
        "Multiple" => ConfigurationType::Multiple,
        other => return Err(format!("unknown config {other:?}").into()),
    })
}

pub(super) fn parse_imported_verified_pairs_file(
    path: &Path,
    image_names: &[String],
) -> Result<Vec<ImportedVerifiedPair>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let name_count: usize = lines
        .next()
        .ok_or("import verified: missing image count")?
        .parse()?;
    if name_count != image_names.len() {
        return Err(format!(
            "import verified: file has {name_count} names, run has {}",
            image_names.len()
        )
        .into());
    }
    for (idx, expected_name) in image_names.iter().enumerate().take(name_count) {
        let file_name = lines
            .next()
            .ok_or("import verified: truncated image name list")?;
        if file_name != expected_name {
            return Err(format!(
                "import verified: name mismatch at {idx}: file {file_name:?} vs run {:?}",
                expected_name
            )
            .into());
        }
    }
    let pair_count: usize = lines
        .next()
        .ok_or("import verified: missing pair count")?
        .parse()?;
    let mut out = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        let head: Vec<&str> = lines
            .next()
            .ok_or("import verified: truncated pair header")?
            .split_whitespace()
            .collect();
        if head.len() < 13 {
            return Err("import verified: pair header needs i j count config e(9)".into());
        }
        let i: usize = head[0].parse()?;
        let j: usize = head[1].parse()?;
        let count: usize = head[2].parse()?;
        let config = parse_config_token(head[3])?;
        let e_vals: Vec<f64> = head[4..13]
            .iter()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()?;
        let essential_matrix = if e_vals.iter().any(|v| v.abs() > 1e-15) {
            Some(Matrix3::from_row_slice(&e_vals))
        } else {
            None
        };
        let mut matches = Vec::with_capacity(count);
        for _ in 0..count {
            let m: Vec<usize> = lines
                .next()
                .ok_or("import verified: truncated correspondence")?
                .split_whitespace()
                .map(|t| t.parse())
                .collect::<Result<_, _>>()?;
            if m.len() != 2 {
                return Err("import verified: correspondence needs qi tj".into());
            }
            matches.push((m[0], m[1]));
        }
        out.push(ImportedVerifiedPair {
            image_i: i,
            image_j: j,
            matches,
            config,
            essential_matrix,
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifiedPairOracle {
    pub(super) inliers: usize,
    pub(super) config: ConfigurationType,
}

pub(super) fn verified_pair_oracle_map(
    imported: &[ImportedVerifiedPair],
) -> HashMap<(usize, usize), VerifiedPairOracle> {
    imported
        .iter()
        .map(|pair| {
            (
                (
                    pair.image_i.min(pair.image_j),
                    pair.image_i.max(pair.image_j),
                ),
                VerifiedPairOracle {
                    inliers: pair.matches.len(),
                    config: pair.config,
                },
            )
        })
        .collect()
}

pub(super) fn verified_pairs_to_pairwise(
    imported: Vec<ImportedVerifiedPair>,
) -> Vec<PairwiseMatches> {
    imported
        .into_iter()
        .map(|p| {
            let essential_matches = match p.config {
                ConfigurationType::Calibrated | ConfigurationType::Multiple
                    if p.essential_matrix.is_some() =>
                {
                    Some(p.matches.clone())
                }
                _ => None,
            };
            PairwiseMatches {
                image_i: p.image_i,
                image_j: p.image_j,
                matches: p.matches,
                two_view_config: Some(p.config),
                essential_matches,
                essential_matrix: p.essential_matrix,
            }
        })
        .collect()
}

/// Read the optional four-column sidecar emitted by `export_features_to_dir`.
/// A missing sidecar is intentional for legacy feature dumps and means that
/// every row remains its own locus.
pub(super) fn read_locus_sidecar(
    path: &Path,
    expected_rows: usize,
) -> Result<Option<Vec<FeatureLocusMetadata>>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!(
                    "{}:{}: invalid locus metadata: {error}",
                    path.display(),
                    line_number + 1
                )
            })?;
        if values.len() != 4 {
            return Err(format!(
                "{}:{}: locus metadata needs x y scale orientation",
                path.display(),
                line_number + 1
            )
            .into());
        }
        rows.push(FeatureLocusMetadata {
            x: values[0],
            y: values[1],
            scale: values[2],
            orientation: values[3],
        });
    }
    if rows.len() != expected_rows {
        return Err(format!(
            "{}: {} metadata rows, expected {} feature rows",
            path.display(),
            rows.len(),
            expected_rows
        )
        .into());
    }
    Ok(Some(rows))
}

/// Parse the compact COLMAP six-column keypoint representation when the
/// descriptor payload has the usual 128 dimensions:
/// `x y a11 a12 a21 a22 d0 ... d127`.  Existing external files use
/// `x y score descriptor...` and take the shared parser path instead.
pub(super) fn read_six_column_locus_features(
    path: &Path,
) -> Result<Option<(FeatureSet, Vec<FeatureLocusMetadata>)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let rows = text
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            (!line.is_empty() && !line.starts_with('#'))
                .then(|| line.split_whitespace().collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    if rows.is_empty() || rows[0].len() != 134 {
        return Ok(None);
    }
    if rows.iter().any(|row| row.len() != 134) {
        return Err(format!(
            "{}: six-column feature rows have inconsistent field counts",
            path.display()
        )
        .into());
    }
    let mut keypoints = Vec::with_capacity(rows.len());
    let mut descriptors = Vec::with_capacity(rows.len());
    let mut metadata = Vec::with_capacity(rows.len());
    for (row_number, row) in rows.iter().enumerate() {
        let parse = |column: usize| {
            row[column].parse::<f64>().map_err(|error| {
                format!(
                    "{} row {}: invalid numeric field {column}: {error}",
                    path.display(),
                    row_number + 1
                )
            })
        };
        let x = parse(0)?;
        let y = parse(1)?;
        let a11 = parse(2)?;
        let a12 = parse(3)?;
        let a21 = parse(4)?;
        let a22 = parse(5)?;
        let descriptor = row[6..]
            .iter()
            .map(|value| {
                value.parse::<f32>().map_err(|error| {
                    format!(
                        "{} row {}: invalid descriptor: {error}",
                        path.display(),
                        row_number + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        keypoints.push(Point2::new(x, y));
        descriptors.push(descriptor);
        metadata.push(FeatureLocusMetadata {
            x,
            y,
            scale: (a11 * a22 - a12 * a21).abs().sqrt(),
            orientation: a21.atan2(a11).rem_euclid(std::f64::consts::TAU),
        });
    }
    Ok(Some((FeatureSet::new(keypoints, descriptors)?, metadata)))
}

/// Read one feature file with the same parser used by the historical batch
/// loader.  The keypoint-only snapshot replay deliberately calls this helper
/// once per file, so the descriptor payload is released before the next image
/// is parsed.
pub(super) fn read_feature_set(path: &Path) -> Result<FeatureSet, Box<dyn std::error::Error>> {
    if let Some((feature_set, _)) = read_six_column_locus_features(path)? {
        return Ok(feature_set);
    }
    Ok(read_external_deep_features_txt(path)?.into_feature_set()?)
}

pub(super) fn list_feature_files(
    dir: &Path,
    feature_suffix: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut files: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(feature_suffix))
        .collect();
    files.sort();
    Ok(files)
}

#[derive(Debug)]
pub(super) struct StreamedVladGlobals {
    pub(super) globals: Option<Vec<Vec<f32>>>,
    pub(super) total_descriptors: usize,
    pub(super) sampled_descriptors: usize,
}

impl StreamedVladGlobals {
    pub(super) fn appearance_globals(&self) -> Result<&[Vec<f32>], &'static str> {
        self.globals
            .as_deref()
            .filter(|globals| !globals.is_empty())
            .ok_or("streamed candidate export requires a nonempty appearance vocabulary; refusing exhaustive fallback")
    }
}

fn count_feature_rows(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .count())
}

fn validate_streamed_candidate_feature(
    image: usize,
    feature_set: &FeatureSet,
    camera: &Camera,
) -> Result<(), String> {
    feature_set
        .validate()
        .map_err(|error| format!("candidate feature image {image} is invalid: {error}"))?;
    for (keypoint, point) in feature_set.keypoints.iter().enumerate() {
        if !point.x.is_finite()
            || !point.y.is_finite()
            || point.x < 0.0
            || point.y < 0.0
            || point.x >= camera.width as f64
            || point.y >= camera.height as f64
        {
            return Err(format!(
                "candidate feature image {image} keypoint {keypoint} ({}, {}) is outside {}x{}",
                point.x, point.y, camera.width, camera.height
            ));
        }
    }
    Ok(())
}

/// Build the historical VLAD vocabulary and per-image globals without ever
/// retaining the local descriptor bank. A cheap row-count prepass fixes the
/// historical global stride; the following two descriptor passes collect the
/// bounded training sample, then aggregate one image at a time.
pub(super) fn stream_vlad_globals_from_feature_files(
    dir: &Path,
    files: &[String],
    rig: &PerImageCameras,
    vocab_size: usize,
) -> Result<StreamedVladGlobals, Box<dyn std::error::Error>> {
    if files.len() != rig.len() {
        return Err(format!(
            "streamed candidate feature/calibration count mismatch: {} files vs {} cameras",
            files.len(),
            rig.len()
        )
        .into());
    }
    let row_counts = files
        .iter()
        .map(|file| count_feature_rows(&dir.join(file)))
        .collect::<Result<Vec<_>, _>>()?;
    let total_descriptors = row_counts.iter().sum::<usize>();
    let stride = (total_descriptors / VLAD_VOCAB_SAMPLE).max(1);
    let mut sample = Vec::<Vec<f32>>::with_capacity(
        total_descriptors
            .div_ceil(stride)
            .min(VLAD_VOCAB_SAMPLE + 1),
    );
    let mut global_row = 0usize;
    for (image, (file, &expected_rows)) in files.iter().zip(&row_counts).enumerate() {
        let feature_set = read_feature_set(&dir.join(file))?;
        validate_streamed_candidate_feature(image, &feature_set, rig.camera(image)?)?;
        if feature_set.descriptors.len() != expected_rows {
            return Err(format!(
                "{} row-count prepass found {expected_rows}, parser found {}",
                dir.join(file).display(),
                feature_set.descriptors.len()
            )
            .into());
        }
        for descriptor in feature_set.descriptors {
            if global_row.is_multiple_of(stride) {
                sample.push(descriptor);
            }
            global_row += 1;
        }
    }
    let sample_refs = sample.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let Some(vocab) = Vocabulary::build(&sample_refs, vocab_size, 10, 0) else {
        return Ok(StreamedVladGlobals {
            globals: None,
            total_descriptors,
            sampled_descriptors: sample.len(),
        });
    };
    drop(sample_refs);
    drop(sample);
    trim_process_allocator();

    // Ordered parallel collection preserves the canonical image/global index
    // mapping. Each worker owns only one local feature set, so peak storage is
    // bounded by the Rayon worker count instead of the image count.
    let globals = files
        .par_iter()
        .enumerate()
        .map(|(image, file)| -> Result<Vec<f32>, String> {
            let feature_set = read_feature_set(&dir.join(file))
                .map_err(|error| format!("cannot read {}: {error}", dir.join(file).display()))?;
            let camera = rig.camera(image).map_err(|error| error.to_string())?;
            validate_streamed_candidate_feature(image, &feature_set, camera)?;
            Ok(vlad(&feature_set.descriptors, &vocab))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StreamedVladGlobals {
        globals: Some(globals),
        total_descriptors,
        sampled_descriptors: total_descriptors.div_ceil(stride),
    })
}
