use std::fs;
use std::path::PathBuf;

use visloc_io::external_deep::{
    parse_external_deep_features_txt, parse_external_deep_matches_txt,
    read_external_deep_features_txt, read_external_deep_matches_txt,
};

fn fixture_dir() -> PathBuf {
    std::env::temp_dir()
        .join("visloc_rs_external_deep_tests")
        .join(std::process::id().to_string())
}

#[test]
fn parses_external_deep_features() {
    let features = parse_external_deep_features_txt(
        r#"
        # X Y SCORE D0 D1 ...
        10.0 20.0 0.91 0.1 0.2 0.3
        30.0 40.0 0.80 1.0 0.0 0.5
        "#,
    )
    .unwrap();

    assert_eq!(features.len(), 2);
    assert_eq!(features.features()[0].xy.x, 10.0);
    assert_eq!(features.features()[1].score, 0.80);
    assert_eq!(features.features()[0].descriptor, vec![0.1, 0.2, 0.3]);

    let feature_set = features.to_feature_set().unwrap();
    assert_eq!(feature_set.keypoints.len(), 2);
    assert_eq!(feature_set.descriptors[1], vec![1.0, 0.0, 0.5]);
}

#[test]
fn parses_external_deep_matches() {
    let matches = parse_external_deep_matches_txt(
        r#"
        # QUERY_IDX TRAIN_IDX CONFIDENCE [DISTANCE]
        0 3 0.99 0.01
        4 8 0.75
        "#,
    )
    .unwrap();

    assert_eq!(matches.len(), 2);
    assert_eq!(matches.matches()[0].query_index, 0);
    assert_eq!(matches.matches()[1].train_index, 8);

    let descriptor_matches = matches.to_descriptor_matches();
    assert_eq!(descriptor_matches[0].distance, 0.01);
    assert_eq!(descriptor_matches[0].confidence, Some(0.99));
    assert!((descriptor_matches[1].distance - 0.25).abs() < 1.0e-6);
}

#[test]
fn reads_external_deep_text_files() {
    let dir = fixture_dir();
    fs::create_dir_all(&dir).unwrap();

    let features_path = dir.join("features.txt");
    let matches_path = dir.join("matches.txt");
    fs::write(&features_path, "10.0 20.0 0.9 0.1 0.2\n").unwrap();
    fs::write(&matches_path, "0 1 0.8\n").unwrap();

    assert_eq!(
        read_external_deep_features_txt(features_path)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        read_external_deep_matches_txt(matches_path).unwrap().len(),
        1
    );
}

#[test]
fn rejects_feature_lines_without_descriptor_values() {
    let error = parse_external_deep_features_txt("10.0 20.0 0.9").unwrap_err();

    assert!(error
        .to_string()
        .contains("invalid external deep feature line"));
}

#[test]
fn rejects_feature_descriptor_dimension_mismatch() {
    let error =
        parse_external_deep_features_txt("10.0 20.0 0.9 0.1 0.2\n30.0 40.0 0.8 1.0").unwrap_err();

    assert!(error.to_string().contains("descriptor 1 has dimension"));
}

#[test]
fn rejects_out_of_range_confidence() {
    let feature_error = parse_external_deep_features_txt("10.0 20.0 1.1 0.1").unwrap_err();
    let match_error = parse_external_deep_matches_txt("0 1 -0.1").unwrap_err();

    assert!(feature_error
        .to_string()
        .contains("invalid external deep feature line"));
    assert!(match_error
        .to_string()
        .contains("invalid external deep match line"));
}

#[test]
fn rejects_non_finite_values() {
    let feature_error = parse_external_deep_features_txt("10.0 20.0 0.9 NaN").unwrap_err();
    let xy_error = parse_external_deep_features_txt("inf 20.0 0.9 0.1").unwrap_err();
    let match_error = parse_external_deep_matches_txt("0 1 0.5 inf").unwrap_err();

    assert!(feature_error
        .to_string()
        .contains("invalid external deep feature line"));
    assert!(xy_error
        .to_string()
        .contains("invalid external deep feature line"));
    assert!(match_error
        .to_string()
        .contains("invalid external deep match line"));
}
