use visloc_rs::{parse_external_deep_features_txt, parse_external_deep_matches_txt};

fn main() {
    let features = parse_external_deep_features_txt(
        r#"
        # X Y SCORE D0 D1 ...
        120.0 140.0 0.99 0.1 0.2 0.3
        260.0 180.0 0.94 1.0 0.0 0.5
        "#,
    )
    .expect("dummy external deep features must parse");

    let matches = parse_external_deep_matches_txt(
        r#"
        # QUERY_IDX TRAIN_IDX CONFIDENCE [DISTANCE]
        0 1 0.97 0.03
        "#,
    )
    .expect("dummy external deep matches must parse");

    let feature_set = features
        .to_feature_set()
        .expect("dummy external deep descriptors must have a consistent dimension");
    let descriptor_matches = matches.to_descriptor_matches();

    println!(
        "external_deep_features={} descriptor_dim={} external_deep_matches={}",
        feature_set.len(),
        feature_set.descriptors.first().map_or(0, Vec::len),
        descriptor_matches.len()
    );
    for descriptor_match in descriptor_matches {
        println!(
            "query={} train={} distance={:.3} confidence={:?}",
            descriptor_match.query_index,
            descriptor_match.train_index,
            descriptor_match.distance,
            descriptor_match.confidence
        );
    }
}
