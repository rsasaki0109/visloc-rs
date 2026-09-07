# Frozen rig BA reference

This bounded diagnostic evaluates the existing Rust-exported
`VISLOC_BA_ORACLE_FIXTURE 1` with Ceres 2.2.0. It is **not a COLMAP mapper
benchmark** and does not change the production Rust solver. The current
checkpoint supports evaluation only; solving and model publication are not
enabled.

Build inside an environment providing Ceres 2.2.0, Eigen and glog:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I/usr/include/eigen3 \
  tools/ceres_rig_reference/ceres_rig_reference.cc -lceres -lglog \
  -o /tmp/ceres_rig_reference
/tmp/ceres_rig_reference --self-test
/tmp/ceres_rig_reference --evaluate-only \
  --fixture /path/to/input.fixture --dump /path/to/new-evaluation.tsv
```

The output parent must already exist. Existing outputs, including symlinks,
are rejected. The evaluator emits every observation's IDs, residuals and
depth, plus the full squared cost (not Ceres's half-cost convention).
Nonfinite or nonpositive-depth observations fail evaluation; none are
silently dropped. Source SHA fields are copied from the fixture, so an
independent audit must verify them against the original source files.

The diagnostic is capped at 512 poses, 8,192 landmarks and 262,144
observations. These are input bounds, not a measured memory guarantee.
Do not enlarge it to the 10k atlas or interpret evaluator timing as BA,
mapper or end-to-end performance. See the
[frozen reference contract](../../docs/openloris_atlas_bounded_ba.md#frozen-ceres-reference-contract-2026-09-08)
for the initial-parity gate and predeclared next-stage solve conditions.
