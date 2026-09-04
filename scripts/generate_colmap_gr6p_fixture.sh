#!/usr/bin/env bash
# Build and run the deterministic COLMAP GR6P oracle generator.
#
# The pinned image is used as the source of the exact COLMAP/PoseLib headers
# and static archives. The image is intentionally runtime-only (it has no
# compiler), so the small fixture program is compiled by the host C++17
# compiler against those extracted artifacts. No package installation or
# network access is needed.
#
# Usage:
#   scripts/generate_colmap_gr6p_fixture.sh
#   scripts/generate_colmap_gr6p_fixture.sh \
#     --output benchmarks/electro/fixtures/colmap-gr6p-v1.json
#
# The default output is under target/ so invoking the script does not replace
# the checked-in fixture. An explicit --output may be used to regenerate it.

set -euo pipefail

image='colmap/colmap@sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f'
default_output='target/colmap_gr6p_fixture.json'
output_path="$default_output"

usage() {
    cat >&2 <<'EOF'
usage: scripts/generate_colmap_gr6p_fixture.sh [--output PATH]

Generate a deterministic COLMAP GR6P fixture using the pinned local image.
The default output is target/colmap_gr6p_fixture.json.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            output_path=$2
            shift 2
            ;;
        -h|--help)
            usage >&1
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname -- "$script_dir")
source_path="$repo_root/benchmarks/electro/tools/colmap_gr6p_fixture.cc"

case "$output_path" in
    /*) ;;
    *) output_path="$repo_root/$output_path" ;;
esac

[ -f "$source_path" ] || {
    echo "generator source not found: $source_path" >&2
    exit 1
}
command -v docker >/dev/null 2>&1 || {
    echo "docker is required (the pinned COLMAP image is not pulled by this script)" >&2
    exit 1
}
cxx_bin=${CXX:-c++}
command -v "$cxx_bin" >/dev/null 2>&1 || {
    echo "host C++17 compiler not found: $cxx_bin" >&2
    exit 1
}
[ -d /usr/include/eigen3/Eigen ] || {
    echo "host Eigen headers not found at /usr/include/eigen3" >&2
    exit 1
}
command -v python3 >/dev/null 2>&1 || {
    echo "python3 is required for JSON validation" >&2
    exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required for provenance output" >&2
    exit 1
}

docker image inspect "$image" >/dev/null 2>&1 || {
    echo "pinned image is not available locally: $image" >&2
    echo "pulling is intentionally disabled; provide the pinned image first" >&2
    exit 1
}

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/colmap-gr6p.XXXXXX")
container_id=''
cleanup() {
    if [ -n "$container_id" ]; then
        docker rm "$container_id" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

# Create (but do not start) the runtime image with the repository mounted
# read-only. docker cp below extracts only pinned build artifacts into tmp_dir;
# the host source remains read-only throughout the build.
container_id=$(docker create --network none --volume "$repo_root:/src:ro" "$image")

mkdir -p \
    "$tmp_dir/usr/local/include/colmap/estimators/solvers" \
    "$tmp_dir/usr/local/include/colmap/geometry" \
    "$tmp_dir/usr/local/include/colmap/util" \
    "$tmp_dir/usr/local/lib" \
    "$tmp_dir/lib"

for header in \
    /usr/local/include/colmap/estimators/solvers/generalized_relative_pose.h \
    /usr/local/include/colmap/geometry/rigid3.h \
    /usr/local/include/colmap/util/types.h \
    /usr/local/include/colmap/util/eigen_alignment.h \
    /usr/local/include/colmap/util/enum_utils.h; do
    docker cp "$container_id:$header" "$tmp_dir$header"
done

for archive in \
    /usr/local/lib/libcolmap_estimators_solvers.a \
    /usr/local/lib/libPoseLib.a \
    /usr/local/lib/libcolmap_math.a; do
    docker cp "$container_id:$archive" "$tmp_dir$archive"
done

# GR6P's COLMAP object contains a fatal-check cold path referring to glog.
# Keep the runtime libraries from the same image so host package versions do
# not affect the oracle binary.
for runtime_library in \
    /usr/lib/x86_64-linux-gnu/libglog.so.0.6.0 \
    /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2; do
    docker cp "$container_id:$runtime_library" "$tmp_dir/lib/"
done
ln -s libglog.so.0.6.0 "$tmp_dir/lib/libglog.so.1"
ln -s libgflags.so.2.2.2 "$tmp_dir/lib/libgflags.so.2.2"

binary="$tmp_dir/colmap_gr6p_fixture"
"$cxx_bin" \
    -std=c++17 -O2 -DNDEBUG -ffp-contract=off \
    -I"$tmp_dir/usr/local/include" -I/usr/include/eigen3 \
    "$source_path" \
    -Wl,--start-group \
    "$tmp_dir/usr/local/lib/libcolmap_estimators_solvers.a" \
    "$tmp_dir/usr/local/lib/libPoseLib.a" \
    "$tmp_dir/usr/local/lib/libcolmap_math.a" \
    -Wl,--end-group \
    -L"$tmp_dir/lib" -Wl,-rpath,"$tmp_dir/lib" \
    -l:libglog.so.0.6.0 -l:libgflags.so.2.2.2 -pthread -ldl -lm \
    -o "$binary"

mkdir -p -- "$(dirname -- "$output_path")"
LD_LIBRARY_PATH="$tmp_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$binary" "$output_path"

python3 - "$output_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    fixture = json.load(handle)
if fixture.get("schema") != "colmap_gr6p_fixture_v1":
    raise SystemExit(f"unexpected fixture schema in {path}")
if fixture.get("solver") != "colmap::GR6PEstimator::Estimate":
    raise SystemExit(f"unexpected solver in {path}")
if fixture.get("num_correspondences") != 6:
    raise SystemExit(f"expected six correspondences in {path}")
if fixture.get("candidate_count", 0) < 1:
    raise SystemExit(f"COLMAP returned no candidates in {path}")
if fixture.get("best_candidate_index") is None:
    raise SystemExit(f"fixture has no best candidate in {path}")
print(
    f"validated {path}: candidates={fixture['candidate_count']} "
    f"best={fixture['best_candidate_index']}"
)
PY

source_sha=$(sha256sum "$source_path" | awk '{print $1}')
fixture_sha=$(sha256sum "$output_path" | awk '{print $1}')
echo "image=$image"
echo "generator_sha256=$source_sha"
echo "fixture_sha256=$fixture_sha"
