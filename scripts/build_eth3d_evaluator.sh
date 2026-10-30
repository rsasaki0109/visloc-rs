#!/usr/bin/env bash
# Clone and build the official ETH3D multi-view evaluation program
# (ETH3DMultiViewEvaluation), which scores point clouds and meshes against
# the high-res training scenes' laser scans.
#
#   scripts/build_eth3d_evaluator.sh [DIR]     # default: ~/visloc_tools
#
# Needs cmake, a C++17 compiler, Boost (filesystem), Eigen 3 and PCL, e.g. on
# Ubuntu: apt install cmake libboost-filesystem-dev libeigen3-dev libpcl-dev.
# The upstream CMakeLists (2021) is patched for current toolchains:
#   - CMake >= 4 refuses its cmake_minimum_required; policy minimum 3.5;
#   - Boost >= 1.89 has no boost_system CMake package (header-only since
#     1.69), so only filesystem is requested;
#   - PCL >= 1.13 needs C++17, but the file adds -std=c++11.
set -euo pipefail
dir=${1:-$HOME/visloc_tools}
mkdir -p "$dir"
cd "$dir"
[ -d multi-view-evaluation ] || git clone -q https://github.com/ETH3D/multi-view-evaluation.git
cd multi-view-evaluation
git log -1 --format="%h %cd"
sed -i "s/COMPONENTS filesystem system/COMPONENTS filesystem/" CMakeLists.txt
sed -i "s/ -std=c++11//" CMakeLists.txt
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. > cmake.log 2>&1 ||
  { tail -20 cmake.log; exit 1; }
make -j"$(nproc)" > make.log 2>&1 || { grep -E "error" make.log | head -20; exit 1; }
echo "built $(pwd)/ETH3DMultiViewEvaluation"
