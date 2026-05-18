#!/usr/bin/env bash
# Phase-{20..27} closeout follow-up — verify binary determinism on a
# fixed configuration.
#
# Background: Phase-26 #4 observed that two `cargo build --release`
# binaries produced from the same source can shift the EuRoC V2_01
# strict rigid ATE by O(10^-3 m), while two runs of the same binary
# are bit-identical. Hypothesis: rustc codegen variation (FMA
# fusion, instruction selection, register allocation) cascading
# through the PnP RANSAC floating-point comparisons.
#
# Mitigation shipped: `rust-toolchain.toml` pins the channel. This
# script quantifies whether the pin eliminates cross-rebuild variance.
#
# Experiment (three-step protocol per variant):
#   1. Clean build → run V2_01 strict.
#   2. Re-run same binary → confirm bit-identical (within-binary
#      determinism sanity).
#   3. `touch` a source file + rebuild → run again → compare.
#
# Two variants are supported via the VARIANT environment variable:
#   - VARIANT=baseline (default): corner extractor, f=3/s=10
#     adaptive-imu-pose. Produces poor tracking on V2_01 (~0.19
#     success-rate, ~4.9 m rigid ATE) but is a clean test of FP
#     reproducibility because it does NOT exercise SuperPoint.
#   - VARIANT=superpoint: SuperPoint+strict-stereo per Phase-26 #1
#     (f=2/s=5, cross-check, pose-prior warm start, local-VI-BA,
#     stereo-bootstrap-strict). This is the config that originally
#     exposed the cross-rebuild variance — the "real" test.
#     Requires pre-exported features at
#     target/euroc_phase26_superpoint/V2_01_easy/cam{0,1}/ (run
#     scripts/export_superpoint_lightglue.py --mono-dir first).
#
# Output: target/binary_determinism_verify_<variant>/{run1,run2,run3}/summary.txt
# plus a top-level COMPARE.md.
#
# Usage:
#   VARIANT=baseline   scripts/verify_binary_determinism.sh
#   VARIANT=superpoint scripts/verify_binary_determinism.sh

set -euo pipefail

EUROC_DIR="${EUROC_DIR:-old_~2026/simple_visual_slam/datasets/euroc}"
SEQ="${SEQ:-V2_01_easy}"
VARIANT="${VARIANT:-baseline}"
OUT_ROOT="${OUT_ROOT:-target/binary_determinism_verify_${VARIANT}}"
MAX_FRAMES="${MAX_FRAMES:-1500}"
SP_DIR="${SP_DIR:-target/euroc_phase26_superpoint}"

cd "$(dirname "$0")/.."

mkdir -p "$OUT_ROOT"

case "$VARIANT" in
    baseline)
        # Phase-25 production-recommended adaptive (f=3/s=10) on the
        # corner extractor. The strict stereo bootstrap is enabled
        # for symmetry with the SuperPoint variant — both fail the
        # bootstrap fairly catastrophically without SuperPoint, but
        # the *FP path* exercised is comparable.
        extra_flags=(
            --motion-model adaptive-imu-pose
            --adaptive-motion-failures-to-switch-to-pose 3
            --adaptive-motion-successes-to-switch-to-imu 10
            --stereo-bootstrap-strict
        )
        ;;
    superpoint)
        cam0_dir="${SP_DIR}/${SEQ}/cam0"
        cam1_dir="${SP_DIR}/${SEQ}/cam1"
        if [ ! -d "$cam0_dir" ] || [ ! -d "$cam1_dir" ]; then
            echo "[determinism] VARIANT=superpoint requires pre-exports at:" >&2
            echo "  $cam0_dir" >&2
            echo "  $cam1_dir" >&2
            echo "Run scripts/export_superpoint_lightglue.py --mono-dir to produce them." >&2
            exit 2
        fi
        # Phase-26 #1 strict-stereo recipe (V-class winner: V1_01
        # rigid ATE 0.0029 m, V2_01 0.0107 m on the original binary).
        # This is the configuration that exposed the cross-rebuild
        # variance Phase-26 #4 reported.
        extra_flags=(
            --gravity 0,0,-9.81
            --cross-check-matcher
            --keyframe-min-translation 0.1
            --max-pose-jump-meters 0.2
            --motion-model adaptive-imu-pose
            --pnp-pose-prior-warm-start
            --vi-init-gyro-std-limit 0.5
            --vi-init-accel-std-limit 5.0
            --vi-init-try-initialize-on-every-frame
            --vi-init-min-stationary-window-seconds 1.5
            --local-vi-ba
            --run-local-vi-ba-at-vi-init-promotion
            --keep-pre-promotion-imu-factors
            --stereo-bootstrap-strict
            --adaptive-motion-failures-to-switch-to-pose 2
            --adaptive-motion-successes-to-switch-to-imu 5
            --feature-extractor superpoint-offline
            --superpoint-features-dir "$cam0_dir"
            --superpoint-cam1-features-dir "$cam1_dir"
        )
        ;;
    *)
        echo "[determinism] unknown VARIANT=$VARIANT (expected baseline|superpoint)" >&2
        exit 2
        ;;
esac

run_one() {
    local label="$1"
    local out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"
    echo "[determinism] $label → $out_dir"
    cargo run --release --features image-io --example euroc_online_slam_vi_image_demo -- \
        --euroc-dir "$EUROC_DIR/$SEQ" \
        --out-dir "$out_dir" \
        --max-frames "$MAX_FRAMES" \
        "${extra_flags[@]}" \
        >"$out_dir/stdout.log" 2>"$out_dir/stderr.log" \
        || { echo "[determinism] run $label FAILED, see $out_dir/stderr.log"; tail -40 "$out_dir/stderr.log"; exit 1; }
}

extract_ate() {
    local summary="$1"
    # Field names match the EuRoC demo's audit-log keys.
    grep -E "^(ate_rigid_rmse_m|ate_similarity_rmse_m|ate_position_rmse_m|ate_orientation_rmse_deg|sim_scale|frames_recorded|tracking_success_rate|map_keyframes|map_landmarks)=" \
        "$summary" 2>/dev/null || echo "(missing $summary)"
}

echo "[determinism] variant=$VARIANT seq=$SEQ"
echo "[determinism] step 1: clean build + first run"
cargo clean --release -q || true
run_one run1

echo "[determinism] step 2: same binary, second run"
run_one run2

echo "[determinism] step 3: touch source + rebuild + run"
touch crates/vision/src/ransac/mod.rs
run_one run3

{
    echo "# Binary determinism verification — VARIANT=$VARIANT"
    echo
    echo "Sequence: $SEQ"
    echo "Toolchain: $(rustc --version)"
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    for label in run1 run2 run3; do
        echo "## $label"
        echo
        echo '```'
        extract_ate "$OUT_ROOT/$label/summary.txt"
        echo '```'
        echo
    done
    echo
    echo "## Interpretation"
    echo
    echo "- run1 vs run2: within-binary determinism. Must be bit-identical."
    echo "- run2 vs run3: cross-rebuild determinism under the pinned toolchain."
    echo "  Any divergence past ~1e-6 m suggests rustc codegen / FMA variation"
    echo "  still propagates through the RANSAC + PnP path even with the pin."
} >"$OUT_ROOT/COMPARE.md"

echo "[determinism] wrote $OUT_ROOT/COMPARE.md"
cat "$OUT_ROOT/COMPARE.md"
