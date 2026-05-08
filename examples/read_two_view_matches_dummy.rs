use visloc_rs::parse_two_view_matches_txt;

fn main() {
    let matches = parse_two_view_matches_txt(
        r#"
        # PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y SCORE
        0 3 120.0 140.0 124.5 141.0 0.99
        1 9 260.0 180.0 263.0 183.5 0.94
        5 12 410.0 220.0 414.0 221.5 0.91
        "#,
    )
    .expect("dummy two-view matches must parse");

    println!("two_view_matches={}", matches.len());
    for feature_match in matches.matches() {
        println!(
            "prev={} curr={} prev_xy=[{:.1}, {:.1}] curr_xy=[{:.1}, {:.1}] score={:?}",
            feature_match.previous_index,
            feature_match.current_index,
            feature_match.previous_xy.x,
            feature_match.previous_xy.y,
            feature_match.current_xy.x,
            feature_match.current_xy.y,
            feature_match.score
        );
    }
}
