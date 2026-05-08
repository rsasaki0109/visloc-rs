#!/usr/bin/env sh
set -eu

examples="
evaluate_trajectory_dummy
evaluate_trajectory_from_kitti_files
evaluate_trajectory_from_tum_files
localize_dummy
localize_colmap_text
localize_colmap_provider
localize_from_files
localize_from_pgm
localize_sequence_from_files
localize_with_corner_extractor
localize_with_extractor_dummy
online_slam_loop_candidate_dummy
online_slam_loop_candidate_with_verifier_dummy
online_slam_pose_graph_loop_demo
online_slam_public_loop_demo
online_slam_pnp_loop_demo
pose_graph_robust_demo
read_two_view_matches_dummy
two_view_match_vo_prior_dummy
track_sequence_dummy
track_sequence_with_extractor_dummy
track_sequence_with_gnss_prior
track_sequence_with_two_view_match_vo_prior
track_sequence_with_visual_odometry_prior
two_view_vo_compare
visual_odometry_prior_dummy
"

for example in $examples; do
    echo "Running example: $example"
    cargo run --example "$example"
done

echo "Running feature-gated example: localize_from_common_image"
cargo run --features image-io --example localize_from_common_image

echo "Running feature-gated example: track_image_sequence_from_common_images"
cargo run --features image-io --example track_image_sequence_from_common_images

echo "Running feature-gated example: track_timestamped_image_sequence_with_gnss_prior"
cargo run --features image-io --example track_timestamped_image_sequence_with_gnss_prior

echo "Running feature-gated example: load_kitti_image_sequence"
cargo run --features image-io --example load_kitti_image_sequence
