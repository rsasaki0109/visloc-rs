use std::path::PathBuf;
use visloc_io::query_features::{parse_query_features_txt, read_query_features_txt};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("query_features")
        .join("query_features.txt")
}

#[test]
fn parses_query_feature_text() {
    let features = parse_query_features_txt(
        r#"
        # X Y D0 D1 ...
        10.0 20.0 0.1 0.2
        30.0 40.0 1.0 2.0
        "#,
    )
    .unwrap();

    assert_eq!(features.len(), 2);
    assert_eq!(features.keypoints[0].x, 10.0);
    assert_eq!(features.keypoints[1].y, 40.0);
    assert_eq!(features.descriptors[0], vec![0.1, 0.2]);
}

#[test]
fn reads_query_feature_text_file() {
    let features = read_query_features_txt(fixture_path()).unwrap();

    assert_eq!(features.len(), 2);
    assert_eq!(features.descriptors[1], vec![1.0, 0.0, 0.5, 0.25]);
}

#[test]
fn rejects_query_feature_lines_without_descriptor_values() {
    let error = parse_query_features_txt("10.0 20.0").unwrap_err();

    assert!(error.to_string().contains("invalid query feature line"));
}

#[test]
fn rejects_query_feature_descriptor_dimension_mismatch() {
    let error = parse_query_features_txt("10.0 20.0 0.1 0.2\n30.0 40.0 1.0").unwrap_err();

    assert!(error.to_string().contains("descriptor 1 has dimension"));
}
