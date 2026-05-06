use std::path::PathBuf;
use visloc_io::descriptors::{parse_landmark_descriptors_txt, read_landmark_descriptors_txt};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("descriptors")
        .join("landmarks.txt")
}

#[test]
fn parses_landmark_descriptor_text() {
    let store = parse_landmark_descriptors_txt(
        r#"
        # LANDMARK_ID D0 D1 ...
        10 0.1 0.2 0.3
        20 1.0 2.0 3.0
        "#,
    )
    .unwrap();

    assert_eq!(store.len(), 2);
    assert_eq!(store.get(10), Some([0.1, 0.2, 0.3].as_slice()));
    assert_eq!(store.get(20), Some([1.0, 2.0, 3.0].as_slice()));
}

#[test]
fn reads_landmark_descriptor_text_file() {
    let store = read_landmark_descriptors_txt(fixture_path()).unwrap();

    assert_eq!(store.len(), 2);
    assert_eq!(store.get(1000), Some([0.1, 0.2, 0.3, 0.4].as_slice()));
    assert_eq!(store.get(1001), Some([1.0, 0.0, 0.5, 0.25].as_slice()));
}

#[test]
fn rejects_descriptor_lines_without_values() {
    let error = parse_landmark_descriptors_txt("10").unwrap_err();
    assert!(error.to_string().contains("invalid descriptor line"));
}
