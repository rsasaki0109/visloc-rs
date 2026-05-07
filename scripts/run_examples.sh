#!/usr/bin/env sh
set -eu

examples="
localize_dummy
localize_colmap_text
localize_colmap_provider
localize_from_files
localize_sequence_from_files
localize_with_extractor_dummy
track_sequence_dummy
track_sequence_with_extractor_dummy
track_sequence_with_gnss_prior
"

for example in $examples; do
    echo "Running example: $example"
    cargo run --example "$example"
done
