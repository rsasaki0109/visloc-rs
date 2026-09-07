# Frozen rig BA reference

This bounded diagnostic evaluates the existing Rust-exported
`VISLOC_BA_ORACLE_FIXTURE 1` with Ceres 2.2.0. It is **not a COLMAP mapper
benchmark** and does not change the production Rust solver. The current
tool has two bounded operations: `--evaluate-only` checks the initial
geometry, and `--solve` runs one fixed Ceres reference solve and publishes a
state report. It does not publish a COLMAP model or change the production Rust
solver. The binary has a compile-time Ceres 2.2.0 guard.

Build inside an environment providing Ceres 2.2.0, Eigen and glog:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I/usr/include/eigen3 \
  tools/ceres_rig_reference/ceres_rig_reference.cc -lceres -lglog \
  -o /tmp/ceres_rig_reference
/tmp/ceres_rig_reference --self-test
/tmp/ceres_rig_reference --evaluate-only \
  --fixture /path/to/input.fixture --dump /path/to/new-evaluation.tsv
/tmp/ceres_rig_reference --solve \
  --fixture /path/to/input.fixture --state /path/to/new-solve.state
```

The output parent must already exist. Existing outputs, including symlinks,
are rejected. The evaluator emits every observation's IDs, residuals and
depth, plus the full squared cost (not Ceres's half-cost convention).
Nonfinite or nonpositive-depth observations fail evaluation; none are
silently dropped. Source SHA fields are copied from the fixture, so an
independent audit must verify them against the original source files.

The solve operation uses the frozen contract: Ceres 2.2.0 AutoDiff PINHOLE
factors, a seven-scalar wxyz-plus-translation pose with
`ProductManifold<QuaternionManifold, EuclideanManifold<3>>`, fixed pose 0,
all points variable, fixed sensor/calibration values, SPARSE_SCHUR with point
group 0 and pose group 1, LM, one thread, initial trust-region radius `1e4`,
and at most 20 iterations. It records the actual options, full Ceres report,
every Ceres iteration, initial/final full squared costs and depth checks, and
all final `POSE`/`LANDMARK` state rows. Ceres's internal half-cost convention
is labeled separately. A state is published only after finite positive-depth,
full-observation validation; output paths are exclusive and existing paths or
symlinks are rejected. The state file starts with
`VISLOC_BA_CERES_SOLVE_STATE 1`, carries all `SOURCE_SHA256_*` fields,
`FIXED_POSE`, counts, initial/final full-cost records and termination/report
records, then writes `POSE id qw qx qy qz tx ty tz` and
`LANDMARK id x y z` rows between explicit state-section markers before
`END`. It is a diagnostic state report, not a publication-ready COLMAP model.

The diagnostic is capped at 512 poses, 8,192 landmarks and 262,144
observations. These are input bounds, not a measured memory guarantee.
Do not enlarge it to the 10k atlas or interpret evaluator timing as BA,
mapper or end-to-end performance. See the
[frozen reference contract](../../docs/openloris_atlas_bounded_ba.md#frozen-ceres-reference-contract-2026-09-08)
for the initial-parity gate and predeclared next-stage solve conditions.

The separate Python publisher consumes a validated state and writes a new
identity-preserving text model. It is a post-solve bridge, not part of the
Ceres solve or the production mapper:

```sh
python3 tools/ceres_rig_reference/publish_model.py \
  --fixture /path/to/input.fixture \
  --state /path/to/solve.state \
  --model /path/to/source/model \
  --rig-manifest /path/to/rig-manifest.txt \
  --out-dir /path/to/new/model
```

All five source SHA256 fields in the fixture and state are checked against
the three source COLMAP files and the rig manifest. The publisher preserves
camera bytes, image/keypoint and point/track identity and order, composes
`T_sensor<-rig * T_rig<-world`, and recomputes each point's mean pixel error
from every observation. It rejects nonpositive depth, changed support,
changed fixed pose, cost/count mismatches, symlink/overlapping inputs, and
existing output paths. Publication is staged and no-clobber; failed
validation does not publish a partial model. The bounded caps are inherited
from the fixture (512 poses, 8,192 points, 262,144 observations).
