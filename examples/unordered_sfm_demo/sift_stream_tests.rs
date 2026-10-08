use super::{
    export_features_to_dir, export_features_to_dir_with_native_keypoints, feature_export_text,
    feature_export_text_with_keypoints, file_fnv1a64, locus_metadata_text,
    sift_stream_manifest_path, stream_export_features_with_loader, validate_sift_stream_manifest,
    write_sift_stream_manifest_atomically, FeatureLocusMetadata,
};
use nalgebra::Point2;
use std::cell::Cell;
use std::fs;
use std::path::PathBuf;
use visloc_rs::FeatureSet;

fn test_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "visloc_sift_stream_{label}_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("thread")
    ));
    let _ = fs::remove_dir_all(&root);
    root
}

fn sample_features(index: usize) -> (FeatureSet, Vec<FeatureLocusMetadata>) {
    let x = 10.0 + index as f64;
    let y = 20.0 + index as f64;
    let feature = FeatureSet::new(
        vec![Point2::new(x, y), Point2::new(x + 1.0, y + 1.0)],
        vec![vec![0.1 + index as f32, 0.2], vec![0.3, 0.4]],
    )
    .unwrap();
    let loci = feature
        .keypoints
        .iter()
        .enumerate()
        .map(|(row, point)| FeatureLocusMetadata {
            x: point.x,
            y: point.y,
            scale: 1.0 + row as f64,
            orientation: 0.25 * row as f64,
        })
        .collect();
    (feature, loci)
}

#[test]
fn stream_export_is_ordered_single_pass_and_byte_identical() {
    let root = test_root("identity");
    let stream_dir = root.join("stream");
    let batch_dir = root.join("batch");
    let mut paths = vec![root.join("b.png"), root.join("a.png")];
    paths.sort();
    let samples = vec![sample_features(0), sample_features(1)];
    let active = Cell::new(0usize);
    let peak = Cell::new(0usize);
    let order = Cell::new(0usize);
    let total = stream_export_features_with_loader(&paths, &stream_dir, |index, path| {
        assert_eq!(index, order.get());
        assert_eq!(path, &paths[index]);
        order.set(order.get() + 1);
        active.set(active.get() + 1);
        peak.set(peak.get().max(active.get()));
        let sample = samples[index].clone();
        active.set(active.get() - 1);
        Ok(sample)
    })
    .unwrap();
    assert_eq!(total, 4);
    assert_eq!(order.get(), 2);
    assert_eq!(active.get(), 0);
    assert_eq!(peak.get(), 1, "the loader must never overlap images");

    let names = vec!["a.png".to_owned(), "b.png".to_owned()];
    let features: Vec<FeatureSet> = samples.iter().map(|(feature, _)| feature.clone()).collect();
    let loci: Vec<Option<Vec<FeatureLocusMetadata>>> =
        samples.iter().map(|(_, loci)| Some(loci.clone())).collect();
    export_features_to_dir(&batch_dir, &names, &features, &loci).unwrap();
    for stem in ["a", "b"] {
        assert_eq!(
            fs::read(stream_dir.join(format!("{stem}_features.txt"))).unwrap(),
            fs::read(batch_dir.join(format!("{stem}_features.txt"))).unwrap()
        );
        assert_eq!(
            fs::read(stream_dir.join(format!("{stem}_loci.txt"))).unwrap(),
            fs::read(batch_dir.join(format!("{stem}_loci.txt"))).unwrap()
        );
        assert!(!stream_dir
            .join(format!(".{stem}_features.txt.tmp"))
            .exists());
        assert!(!stream_dir.join(format!(".{stem}_loci.txt.tmp")).exists());
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn stream_failure_keeps_existing_output_and_writes_no_partial_file() {
    let root = test_root("failure");
    let output_dir = root.join("features");
    fs::create_dir_all(&output_dir).unwrap();
    let existing = output_dir.join("a_features.txt");
    fs::write(&existing, b"previous-complete-file\n").unwrap();
    let paths = vec![root.join("a.png"), root.join("b.png")];
    let error = stream_export_features_with_loader(&paths, &output_dir, |index, _| {
        if index == 0 {
            return Err("synthetic extraction failure".into());
        }
        Ok(sample_features(index))
    })
    .unwrap_err();
    assert!(error.to_string().contains("synthetic extraction failure"));
    assert_eq!(fs::read(&existing).unwrap(), b"previous-complete-file\n");
    assert!(!output_dir.join("b_features.txt").exists());
    assert!(!output_dir.join("a_loci.txt").exists());
    assert!(!output_dir.join(".a_features.txt.tmp").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn native_coordinate_export_reuses_canonical_descriptor_rows() {
    let root = test_root("native-coordinate-export");
    let output_dir = root.join("features");
    let names = vec!["image.png".to_owned()];
    let features = vec![FeatureSet::new(
        vec![Point2::new(10.0, 20.0), Point2::new(30.0, 40.0)],
        vec![vec![0.1, 0.2], vec![0.3, 0.4]],
    )
    .unwrap()];
    let native_keypoints = vec![vec![Point2::new(100.0, 200.0), Point2::new(300.0, 400.0)]];
    let loci = vec![None];
    export_features_to_dir_with_native_keypoints(
        &output_dir,
        &names,
        &features,
        &native_keypoints,
        &loci,
    )
    .unwrap();
    let exported = fs::read_to_string(output_dir.join("image_features.txt")).unwrap();
    assert_eq!(
        exported,
        feature_export_text_with_keypoints(&native_keypoints[0], &features[0].descriptors)
    );
    assert!(exported.contains("100.000000 200.000000"));
    assert!(exported.contains("0.100000 0.200000"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn resume_manifest_requires_matching_config_source_and_outputs() {
    let root = test_root("resume-manifest");
    let source = root.join("source.png");
    let output_dir = root.join("features");
    fs::create_dir_all(&output_dir).unwrap();
    fs::write(&source, b"source-image-bytes").unwrap();
    let (features, loci) = sample_features(3);
    let feature_path = output_dir.join("source_features.txt");
    let loci_path = output_dir.join("source_loci.txt");
    fs::write(&feature_path, feature_export_text(&features)).unwrap();
    fs::write(&loci_path, locus_metadata_text(&loci)).unwrap();
    let source_digest = file_fnv1a64(&source).unwrap();
    let feature_digest = file_fnv1a64(&feature_path).unwrap();
    let loci_digest = file_fnv1a64(&loci_path).unwrap();
    let manifest = sift_stream_manifest_path(&output_dir, "source");
    write_sift_stream_manifest_atomically(
        &manifest,
        0x1234,
        source_digest,
        features.len(),
        feature_digest,
        loci_digest,
    )
    .unwrap();
    assert_eq!(
        validate_sift_stream_manifest(&manifest, 0x1234, source_digest, &feature_path, &loci_path,)
            .unwrap(),
        Some(features.len())
    );

    let mut tampered = fs::read(&feature_path).unwrap();
    let last = tampered.len() - 1;
    tampered[last] = if tampered[last] == b'\n' { b' ' } else { b'\n' };
    fs::write(&feature_path, tampered).unwrap();
    assert_eq!(
        validate_sift_stream_manifest(&manifest, 0x1234, source_digest, &feature_path, &loci_path,)
            .unwrap(),
        None
    );
    fs::write(&feature_path, feature_export_text(&features)).unwrap();
    assert!(validate_sift_stream_manifest(
        &manifest,
        0x9999,
        source_digest,
        &feature_path,
        &loci_path,
    )
    .unwrap()
    .is_none());
    assert!(validate_sift_stream_manifest(
        &manifest,
        0x1234,
        (source_digest.0 + 1, source_digest.1),
        &feature_path,
        &loci_path,
    )
    .unwrap()
    .is_none());
    let mut changed_source = fs::read(&source).unwrap();
    changed_source[0] ^= 1;
    fs::write(&source, changed_source).unwrap();
    let changed_source_digest = file_fnv1a64(&source).unwrap();
    assert!(validate_sift_stream_manifest(
        &manifest,
        0x1234,
        changed_source_digest,
        &feature_path,
        &loci_path,
    )
    .unwrap()
    .is_none());
    assert!(!manifest
        .with_file_name(format!(
            ".{}.tmp",
            manifest.file_name().unwrap().to_string_lossy()
        ))
        .exists());
    let _ = fs::remove_dir_all(root);
}
