use std::path::PathBuf;

use nalgebra::Point2;
use visloc_io::two_view_matches::{parse_two_view_matches_txt, read_two_view_matches_txt};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("two_view_matches")
        .join("matches.txt")
}

#[test]
fn parses_two_view_match_text() {
    let matches = parse_two_view_matches_txt(
        r#"
        # PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y [SCORE]
        0 4 10.0 20.0 11.5 20.5 0.98
        1 8 30.0 40.0 31.0 41.0
        "#,
    )
    .unwrap();

    assert_eq!(matches.len(), 2);
    assert_eq!(matches.matches()[0].previous_index, 0);
    assert_eq!(matches.matches()[0].current_index, 4);
    assert_eq!(matches.matches()[0].previous_xy, Point2::new(10.0, 20.0));
    assert_eq!(matches.matches()[0].current_xy, Point2::new(11.5, 20.5));
    assert_eq!(matches.matches()[0].score, Some(0.98));
    assert_eq!(matches.matches()[1].score, None);
    assert_eq!(
        matches.matched_previous_keypoints(),
        vec![Point2::new(10.0, 20.0), Point2::new(30.0, 40.0)]
    );
    assert_eq!(
        matches.matched_current_keypoints(),
        vec![Point2::new(11.5, 20.5), Point2::new(31.0, 41.0)]
    );
}

#[test]
fn reads_two_view_match_text_file() {
    let matches = read_two_view_matches_txt(fixture_path()).unwrap();

    assert_eq!(matches.len(), 3);
    assert_eq!(matches.matches()[2].previous_index, 7);
    assert_eq!(matches.matches()[2].score, Some(0.77));
}

#[test]
fn rejects_invalid_two_view_match_lines() {
    let error = parse_two_view_matches_txt("0 1 10.0 20.0 11.0").unwrap_err();

    assert!(error.to_string().contains("invalid two-view match line"));
}
