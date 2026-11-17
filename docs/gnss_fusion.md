# GNSS + Visual-Odometry Fusion

`visloc_slam::gnss_fusion` fuses a visual-odometry (VO / VIO) trajectory with a
GNSS fix stream in **one joint optimization**: VO relative-pose factors and
GNSS position factors are solved together with the GNSS-to-map alignment,
instead of using each fix only as a per-frame search prior (which is what
[`track_sequence_with_gnss_prior`](gnss_demo.md) does).

It is opt-in: none of the existing GNSS-prior examples or their check scripts
change behaviour.

## What it is (and is not)

- **Loosely coupled in the GNSS domain.** The measurement is the receiver's
  position solution (ENU metres plus a 3×3 covariance or horizontal/vertical
  accuracies). Raw pseudoranges, Doppler, carrier phase, satellite geometry,
  clock states, and RTK ambiguity resolution are **not** modelled.
- **Pose-graph fusion on the visual side.** VO enters as relative-pose
  factors between consecutive frames, not as image reprojection residuals, so
  this is not a visual–GNSS bundle adjustment. Landmarks are untouched.
- **Jointly estimated alignment.** The ENU-to-map similarity is an optimization
  variable alongside every frame pose, not a fixed pre-alignment.
- **No IMU.** There is no INS mechanization or IMU preintegration here; a VIO
  front end can feed its (gravity-aligned) trajectory in as the VO input.

## Model

States, all in the VO **map** frame:

| State | DoF | Notes |
| --- | --- | --- |
| Frame pose `(Rᵢ, pᵢ)` | 6 | camera-to-map; the oldest frame of the window is held fixed (gauge) |
| Frame log-scale `σᵢ` | 1 | only in `GnssScaleMode::ScaleDrift` (monocular scale drift, Sim(3)-style node) |
| Alignment `x_enu = s·R·x_map + t` | 4–7 | yaw + translation (`AlignmentRotationDof::YawOnly`), or full rotation (`Full`); `s` fixed to 1 in `Metric`, free otherwise |

Factors:

| Factor | Residual | Weighting |
| --- | --- | --- |
| VO relative pose `Zᵢⱼ = Tᵢ⁻¹Tⱼ` | `[e^{-σᵢ}Rᵢᵀ(pⱼ−pᵢ) − t_Z; Log(R_Zᵀ Rᵢᵀ Rⱼ); σⱼ − σᵢ]` | `VoNoiseModel`: translation sigma = floor + ratio·step length, rotation sigma per step, log-scale random walk |
| GNSS position | `s·R·pᵢ + t + R·Rᵢ·l − z` | per-fix 3×3 information; robust kernel or GNC; `l` = antenna lever arm in the camera frame |
| Alignment prior | `[t − t₀; Log(R R₀ᵀ); ln s − ln s₀]` | sliding window only (`AlignmentPriorSigmas`) |

All Jacobians are analytic and unit-tested against central finite differences
(`pipelines/slam/src/gnss_pose_graph.rs`).

The solver is Levenberg–Marquardt with Marquardt (`diag(H)`) damping on the
crate's block-sparse Cholesky (7×7 blocks; nodes in time order with the
alignment block last, so a VO chain plus GNSS factors forms an "arrow" matrix
that factors without fill). It reuses the SLAM crate's existing machinery:
`RobustKernel` (Huber/Cauchy IRLS), `gnc::GncState` (Graduated Non-Convexity),
and `block_cholesky`.

### Pipeline

1. **Time synchronization** — `visloc_fusion::interpolate_gnss_fix` resamples
   the receiver's stream (its own rate and clock offset) at each camera frame
   time. A fix within `max_nearest_offset` (20 ms) is used directly; otherwise
   the two bracketing fixes are linearly interpolated if they are at most
   `max_bracket_gap` (1.5 s) apart. Wider gaps are **dropouts**: those frames
   get no GNSS factor and are carried by the VO chain.
2. **Per-fix covariance** — the fix's full `PositionCovariance` if present,
   else its horizontal/vertical accuracies (vertical defaults to 2× horizontal),
   else `default_horizontal_sigma` / `default_vertical_sigma`. Interpolation
   blends the two covariances *convexly* (GNSS errors are time-correlated, so
   the midpoint is not treated as more accurate than either fix) and adds
   `motion_sigma_per_second` × distance-to-nearest-fix. When the camera is
   faster than the receiver, one fix feeds several frames; its information is
   split between them (`share_fix_information`) so it is not double counted.
3. **Alignment bootstrap** — RANSAC (2-point minimal samples for yaw-only,
   3-point for full rotation) plus a weighted Procrustes/Umeyama refit over the
   earliest fixes that span `min_horizontal_extent` (30 m), where VO drift is
   smallest. The lever arm is folded in by fixed-point iteration. Until the
   extent is reached the alignment is unobservable and `optimize` returns
   `GnssFusionError::AlignmentNotObservable`.
4. **Joint solve** — with the default `GnssRobustMode::Gnc` (truncated least
   squares, inlier scale at the χ²(3) 99.9 % gate ≈ 4.03σ): a least-squares
   warm start, GNC annealing, χ² classification of every GNSS factor, then a
   final refinement with the rejected fixes removed (`refine_without_outliers`).
   `GnssRobustMode::Kernel(Huber/Cauchy)` and `GnssRobustMode::None` are also
   available.

### Batch vs sliding window

- `GnssVoFusionConfig::window = None` (default): one full-batch solve over all
  frames, frame 0 fixed.
- `window = Some(n)`: a causal fixed-lag smoother. Frames and fixes are pushed
  in time order; every `window_update_stride` frames the last `n` frames are
  re-solved with the oldest one fixed at its current estimate and the previous
  alignment as a Gaussian prior. New frames are initialized by chaining VO
  motion onto the latest fused state. Each frame's reported ENU pose is the
  one from the last solve that contained it (the alignment keeps absorbing VO
  drift after a frame leaves the window). There is no marginalization: what
  dropped frames said about the alignment survives only through that prior.

## API

```rust
use visloc_rs::slam::gnss_fusion::{
    fuse_gnss_with_visual_odometry, GnssVoFusion, GnssVoFusionConfig,
};

// vo: Vec<TimedPose> (world-to-camera in the VO map frame, increasing time)
// gnss: MeasurementBuffer<GnssMeasurement> (ENU antenna positions)
let mut config = GnssVoFusionConfig::metric_gravity_aligned(); // or ::monocular()
config.lever_arm = nalgebra::Vector3::new(0.0, -1.5, -0.8); // antenna in camera frame
let fused = fuse_gnss_with_visual_odometry(&vo, &gnss, &config)?;
// fused.trajectory_enu, fused.alignment, fused.bootstrap,
// fused.frames[i].gnss (Inlier / Outlier / NoFix), fused.summary

// Incremental use:
let mut fusion = GnssVoFusion::new(config);
fusion.push_vo_pose(&vo[0])?;
fusion.push_gnss(fix);
let update = fusion.optimize()?; // bootstraps on first success
let enu = fusion.enu_trajectory();
```

Lower-level pieces, usable on their own:

- `visloc_fusion::{interpolate_gnss_fix, gnss_fix_covariance, GnssInterpolationConfig}` — time sync.
- `visloc_slam::{GnssPoseGraph, VoRelativeFactor, GnssPositionFactor, GnssAlignmentPrior, GnssPoseGraphConfig}` — the optimizer.
- `visloc_slam::{bootstrap_gnss_alignment, fit_gnss_alignment}` — robust / closed-form alignment.
- `visloc_slam::gnss_synthetic` — deterministic synthetic scenarios.
- `gnss_fusion::{position_rmse, aligned_position_rmse, transform_trajectory}` — evaluation helpers.
- `visloc_io::kitti_raw_gnss::{LocalEnuFrame, kitti_oxts_gnss_measurements, KittiRawStereoCalibration}` —
  KITTI raw OXTS to ENU positions / IMU poses / GNSS fixes, and the raw IMU → camera calibration chain.
- `GnssVoFusionConfig::gate_at_fix_accuracy` (opt-in, default off) — gate each
  fix at its own accuracy rather than the shared one; see "Results on KITTI raw".
- `GnssPoseGraphConfig::vo_robust` (`config.optimizer.vo_robust`, opt-in,
  default `RobustKernel::None`) — a Huber or Cauchy kernel on the VO
  relative-pose factors, applied by IRLS in every LM stage, so a corrupted VO
  step is down-weighted instead of correct fixes being rejected; see "Robust
  VO factors". `GnssPoseGraph::relative_squared_residuals` reports the VO
  residuals.

Choose the alignment/scale model to match the VO:

| VO | `rotation_dof` | `scale_mode` | Preset |
| --- | --- | --- | --- |
| Stereo VO / VIO, map frame gravity aligned with `+z` up | `YawOnly` | `Metric` | `metric_gravity_aligned()` |
| Metric VO, map frame = first camera frame | `Full` | `Metric` | — |
| Monocular VO | `Full` | `ScaleDrift` (or `GlobalScale`) | `monocular()` |

## Run It

```bash
cargo run --example gnss_vo_fusion_demo
cargo run --example gnss_vo_fusion_demo -- --scenario monocular
cargo run --example gnss_vo_fusion_demo -- --window 150
cargo run --example gnss_vo_fusion_demo -- --robust none   # see outliers pull LS
cargo run --example gnss_vo_fusion_demo -- --out-dir target/visloc_gnss_fusion_demo
sh scripts/check_gnss_fusion_demo_outputs.sh
```

The synthetic drive (`SyntheticGnssVoConfig::default()`): a ~1 km loop over
120 s, 600 camera frames at 5 Hz; VO with ~1°/min heading drift, 2 % scale
error, and per-step noise; a 2 Hz receiver with a 37 ms clock offset,
0.8 m / 1.5 m (horizontal / vertical) noise, a 15 s dropout, and three
multipath bursts (6 fixes offset by 21–28 m); antenna lever arm 1.5 m up and
0.8 m behind the camera. `--out-dir` writes `summary.json` and a per-frame
`trajectory.csv` (truth, VO, fused, GNSS status).

Results from the deterministic scenarios (debug build):

| Scenario | VO only, bootstrap-aligned | VO only, oracle-aligned to truth | Fused | Notes |
| --- | --- | --- | --- | --- |
| Metric, batch, GNC | 5.25 m | 2.24 m (SE(3)) | **0.90 m** | dropout frames 1.70 m; all 19 frames touching corrupted fixes rejected, none elsewhere; yaw error 0.75° |
| Metric, batch, `--robust none` | 5.25 m | 2.24 m | 1.33 m | frames near outliers 1.68 m RMSE vs 0.54 m with GNC |
| Metric, sliding window 150 | 5.25 m | 2.24 m | **1.04 m** | causal; 60 window solves |
| Monocular, batch, GNC | 185 m | 3.22 m (Sim(3)) | **0.37 m** | alignment scale error −2.2 %, rotation error 0.7° |

"Oracle-aligned" is the best-fit rigid (or similarity) transform of the VO onto
ground truth, i.e. the most favourable possible VO-only number; the fused
trajectory is evaluated with no alignment at all. In the metric case the
residual fused error is dominated by the unmodelled 2 % VO scale error, which
the metric model cannot absorb (the monocular `ScaleDrift` model can, hence its
lower error).

Runtime: the full-batch 600-frame demo takes ~16 s in an unoptimized debug
build (≈40 LM iterations including GNC); release builds are one to two orders
of magnitude faster.

## Results on KITTI raw

The first run on real data: three KITTI raw drives with their OXTS RT3003
GPS/INS logs (2026-11).

### Setup

- **Drives** (`2011_09_30`, the residential/city drives behind KITTI odometry
  05/06/07; all three revisit places):

  | Drive | Odometry seq | Frames used | Path | Duration |
  | --- | --- | --- | --- | --- |
  | `2011_09_30_drive_0027` | 07 | 1,106 (10 Hz) | 695 m | 114 s |
  | `2011_09_30_drive_0020` | 06 | 1,104 (10 Hz) | 1,236 m | 114 s |
  | `2011_09_30_drive_0018` | 05 | 1,381 (every 2nd of 2,762, 5 Hz) | 2,207 m | 287 s |

  0018 runs at stride 2 only because the VO demo holds all frames in memory
  (~5 GB RSS per 1,100 frames), which did not fit next to other jobs.
- **Ground truth**: the synced OXTS packets (`oxts/data`, 10 Hz, RTK: the
  reported `pos_accuracy` has a median of 6 / 1.3 / 2 cm on 0027 / 0020 /
  0018), converted exactly (WGS84 → ECEF → ENU at the first packet, with the
  meridian convergence applied to the attitude) by the new
  `visloc_io::kitti_raw_gnss`. The camera pose is
  `T_enu_imu · (R_rect_00 · T_cam0_velo · T_velo_imu)⁻¹`, and the lever arm
  (the OXTS reference point in the rectified camera-0 frame) is
  (−0.32, 0.73, −1.14) m. Frames and packets are paired by index (the sync
  timestamps differ by a few ms). With the native fixes, the fused camera
  positions match this ground truth to typically 5–10 cm away from VO
  failures, which checks the calibration chain and the lever arm.
- **VO**: `online_slam_stereo_vo_kitti_demo --frontend deep --no-stereo-ba`
  (the pure-Rust HOG-like + mutual-softmax frontend with the
  `run_kitti_deep_vo_smoke.sh` defaults; no BA, no loop closure) on the
  rectified `image_00` / `image_01`, plus
  `--motion-scale-rescue-min-translation-ratio 0` (see the negative results).
  This is **not** the SuperPoint/LightGlue + BA + loop-closure stack of
  [`kitti_multiseq_benchmark.md`](kitti_multiseq_benchmark.md) (1.4–2.3 m on
  these sequences), which needs a torch export; its drift here is 9–10 m.
- **Degraded GNSS** (`--gnss degraded`, the default of
  `examples/kitti_raw_gnss_vo_fusion.rs`): the OXTS positions subsampled to
  1 Hz; per-axis noise σ = 2 m east/north and 4 m up, of which 80 % of the
  variance is a first-order Gauss–Markov process with τ = 45 s and 20 % white;
  multipath bursts of 1–3 fixes offset 10–30 m horizontally (vertical
  N(0, (offset/3)²)) started with probability 1.5 % per fix (≈3 % of fixes);
  a 20 s dropout starting halfway through the drive; every fix reports
  2 m / 4 m accuracy. SplitMix64 + Box–Muller from `--seed` (1, 2, 3).
- **Native GNSS** (`--gnss native`): the same OXTS positions at 1 Hz with their
  own `pos_accuracy`. Since they are also the ground truth, this run measures
  only how well the fusion follows near-perfect fixes through the VO.
- **Fusion**: `GnssVoFusionConfig::metric_gravity_aligned()` with
  `rotation_dof = Full` (the VO map is the first camera frame, not gravity
  aligned), GNC, default VO noise model, the lever arm above; sliding window =
  30 s (300 frames at 10 Hz, 150 at 5 Hz), update stride 10.
- **Metrics**: RMS of camera-centre error vs OXTS ground truth. VO alone after
  its best-fit SE(3) alignment onto the ground truth (oracle); raw GNSS = RMS of
  the fixes against the true antenna (OXTS) position, no alignment; fused = the
  ENU trajectory, no alignment.

### Results (seed 1, default fusion)

| Drive | VO alone, SE(3)-aligned | Raw degraded GNSS (3-D / horizontal) | Fused batch (3-D / horizontal) | Fused sliding window (3-D / horizontal) | Fused, native OXTS fixes (batch / window) |
| --- | --- | --- | --- | --- | --- |
| 0027 (07) | 9.22 m | 5.59 / 4.39 m | **3.74** / 2.69 m | 3.83 / 2.82 m | 0.61 / 0.87 m |
| 0020 (06) | 9.79 m | 5.62 / 4.39 m | **4.59** / 3.77 m | 6.24 / 3.10 m | 0.16 / 0.17 m |
| 0018 (05) | 10.06 m | 5.78 / 4.18 m | **4.03** / 2.20 m | 4.67 / 2.91 m | 0.09 / 0.09 m |

Mean over seeds 1–3 (fused 3-D ATE; "gate" = the new opt-in
`gate_at_fix_accuracy`, "gravity" = `--alignment gravity-yaw`):

| Drive | Raw GNSS (3-D / horiz.) | Batch | Batch + gate | Window | Window + gate | Window + gate + gravity | Multipath-adjacent frames flagged (batch: default → gate) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0027 | 6.55 / 5.73 m | 4.08 m | **3.56 m** | 4.42 m | 4.09 m | 4.10 m | 0 → 112 of 151 |
| 0020 | 6.56 / 5.72 m | 4.76 m | **4.33 m** | 6.05 m | 5.84 m | **4.24 m** | 1 → 118 of 149 |
| 0018 | 5.97 / 4.81 m | 3.50 m | **3.38 m** | 4.04 m | 3.80 m | **3.71 m** | 26 → 121 of 160 |

Frames flagged away from injected multipath: 0 in every batch run; 24 in one
run (0020, seed 1, window + gate without gravity), 0 elsewhere.

What this shows:

- **Batch fusion beats both inputs on every drive and seed**: seed means of
  3.4–4.8 m against 9–10 m for the oracle-aligned VO and 6.0–6.6 m for the
  raw fixes. The default sliding window does not always: on 0020 it is worse
  than the raw fixes in 3-D for seeds 1 and 3 (6.24 vs 5.62 m, 6.69 vs
  6.23 m). The gain
  over raw GNSS is mostly horizontal (e.g. 5.73 → 2.94 m on 0027 with the
  gate); the vertical error stays close to the receiver's correlated vertical
  error, which the VO cannot observe.
- **The gain is far smaller than on the synthetic drive** (5.25 → 0.90 m).
  The synthetic receiver has white noise; here 80 % of the error is
  correlated over 45 s, so averaging over ~100–300 fixes removes little of
  it, and the fusion still treats the fixes as independent.
- **Batch beats the sliding window** on all three drives. With a full
  rotation the 30 s window lets roll and pitch wander: on 0020 the window's
  horizontal error is lower than batch, but its 3-D error is 6.05 m. Levelling
  the VO with the first frame's roll/pitch (an accelerometer stand-in) and
  estimating yaw only fixes that (4.24 m). For batch it changes nothing.
- **Multipath was mostly not rejected by default** (see the defect below). With
  the gate, 70–80 % of the frames next to a jump are flagged, and batch ATE
  improves on every drive and seed except 0020 seed 3 (+0.04 m).
- Runtime (release, 4-core shared machine): VO 0.3–0.6 s per frame,
  single-threaded; fusion batch 0.1–0.9 s, sliding window 0.8–4.5 s per drive.

### Negative results and defects found

- **Information sharing hid multipath from the gate.** With a 10 Hz camera
  and a 1 Hz receiver each fix feeds ~10 frames, and
  `share_fix_information` multiplies each frame's covariance by ~10. The χ²
  gate and GNC then judge each frame at √10 ≈ 3.2× the fix sigma, so a 20 m
  jump against 2 m sigmas passes as an inlier (0 of 151 adjacent frames
  flagged on 0027). The new opt-in `GnssVoFusionConfig::gate_at_fix_accuracy`
  (backed by `GnssPoseGraph::gnss_gate_scales`) judges consistency at the
  unshared fix accuracy while keeping the shared weighting. It is neutral on
  the synthetic drive (0.898 → 0.897 m, 20 instead of 19 frames flagged,
  none elsewhere) and helps batch on KITTI. It stays off by default until it
  has been measured on more data.
- **VO failures are not robustified; correct GNSS gets rejected instead.**
  GNC acts only on GNSS factors, and the VO noise model is a fixed, confident
  per-step sigma. On 0027 a truck crossing in front of the camera (frames
  ~615–640, the event analysed in
  [`kitti_multiseq_benchmark.md`](kitti_multiseq_benchmark.md)) corrupts the
  VO rotation. Even with near-perfect native fixes, 17 of those frames' fixes
  are rejected and the error peaks at 4.9 m (0.61 m RMS against 0.09–0.16 m
  on the other drives). The opt-in robust VO kernel fixes this case (next
  section).
- **A VO frontend lock-in on deceleration.** With the default motion-scale
  rescue, the deep frontend on 0020 clamps every step to 1.57 m from frame
  ~790 on while the car slows to 0.7–1 m per frame (raw PnP had it right). Each
  rescued step feeds the history the rescue compares against, so it never
  releases: 308 of 1,103 pairs were rescued. The VO reached 30.2 m ATE. Fusing
  that VO gave 24.4 m (max 104 m) with degraded GNSS and 13.3 m (max 74 m)
  even with the native fixes, because the fixes were rejected as outliers.
  `--motion-scale-rescue-min-translation-ratio 0` turns off that collapse
  branch. It is used for all three drives above. On 0027 the default run
  fired no rescue at all, so its VO is unchanged; 0018 was run only with the
  flag. The frontend default is not changed here.
- **Levelling the VO with gravity (yaw-only alignment) does not help batch**
  (±0.04 m), only the sliding window.

### Robust VO factors (`vo_robust`, 2026-11-17)

`--vo-robust cauchy:<c>|huber:<delta>` (`GnssPoseGraphConfig::vo_robust`)
puts a robust kernel on the VO factors' whitened squared residual. It is the
same IRLS as the GNSS kernel, in every LM stage, including GNC's levels and
the final refinement. Same VO runs as above (0020 with the rescue off, 0027),
batch / window ATE in metres:

| Drive, GNSS | Plain | Cauchy c = 3 | Cauchy c = 5 | Huber δ = 3 |
| --- | ---: | ---: | ---: | ---: |
| 0027, native fixes | 0.61 / 0.87 (17 fixes rejected, max 4.91) | **0.18 / 0.19** (0 rejected, max 3.59) | 0.17 / 0.19 (0, max 3.55) | 0.23 / 0.58 (0, max 2.46) |
| 0020, native fixes | 0.16 / 0.17 (3 rejected, max 1.21) | 0.15 / 0.16 (0, max 1.79) | 0.15 / 0.15 (0, max 1.71) | 0.16 / 0.17 (3, max 1.21) |
| 0027, degraded, seeds 1–3 mean | 4.08 / 4.42 | 4.08 / 4.42 | 4.08 / 4.42 | 4.08 / 4.42 |
| 0020, degraded, seeds 1–3 mean | 4.76 / 6.05 | 4.76 / 6.05 | 4.76 / 6.05 | 4.76 / 6.05 |

- **It fixes the truck crossing.** With near-perfect fixes, 0027 goes from
  0.61 to 0.18 m, and no correct fix is rejected any more.
- **With degraded GNSS nothing changes** to two decimals. The receiver error
  (2 m, correlated over 45 s) dwarfs the VO residuals, so no VO factor
  reaches the kernel's knee.
- **On 0020 the max error grows** (1.21 → 1.79 m with Cauchy 3), although the
  RMS improves slightly. Huber 3 leaves 0020 as it is.
- It stays off by default.
- `tests/gnss_fusion.rs::robust_vo_factors_absorb_a_corrupted_vo_step`
  corrupts one synthetic VO step (6° yaw and 5 m). Plain fusion gives 1.58 m,
  with Cauchy 3 it gives 0.77 m.

### Reproduce

```sh
# Data (KITTI raw, ~0.5 GB per 1,100-frame drive: only image_00/image_01/oxts
# and the date's calibration are fetched, by HTTP range requests):
#   https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip
#   https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_00XX/2011_09_30_drive_00XX_sync.zip
for d in 0027 0020 0018; do
  python3 scripts/fetch_kitti_raw_drive.py --drive 2011_09_30_drive_$d \
      --out ~/data/kitti_raw/2011_09_30_drive_$d --skip-tracklets --skip-extract-oxts
done
# VO + fusion (degraded seed 1 and native), outputs in target/kitti_raw_gnss/<drive>/:
scripts/run_kitti_raw_gnss_fusion_eval.sh --drive 2011_09_30_drive_0027 \
    --data-root ~/data/kitti_raw/2011_09_30_drive_0027
scripts/run_kitti_raw_gnss_fusion_eval.sh --drive 2011_09_30_drive_0020 \
    --data-root ~/data/kitti_raw/2011_09_30_drive_0020
scripts/run_kitti_raw_gnss_fusion_eval.sh --drive 2011_09_30_drive_0018 \
    --data-root ~/data/kitti_raw/2011_09_30_drive_0018 --vo-stride 2
# Variants (seeds 2/3, gate, gravity levelling) on an existing VO run:
target/release/examples/kitti_raw_gnss_vo_fusion \
    --date-dir ~/data/kitti_raw/2011_09_30_drive_0027/2011_09_30 \
    --drive 2011_09_30_drive_0027 \
    --vo-poses target/kitti_raw_gnss/2011_09_30_drive_0027/vo/vo_poses.txt \
    --window 300 --seed 2 --gate-at-fix-accuracy --alignment gravity-yaw
```

Each run writes `summary.json` (all metrics and the noise parameters),
`trajectory.csv` (truth, SE(3)-aligned VO, batch and window ENU positions with
per-frame GNSS status) and `gnss.csv` (fix, true antenna position, multipath
flag). Once the VO has run, the fusion needs only `oxts/`,
`image_00/timestamps.txt` and the calibration files. The fusion's own RANSAC
seed is the default `GnssBootstrapConfig::seed`.

## Tests

- `pipelines/slam/src/gnss_pose_graph.rs` unit tests: finite-difference checks
  of the GNSS, VO-relative, and alignment-prior Jacobians; RANSAC bootstrap
  recovery of yaw/scale/translation with outliers and of a full rotation;
  short-baseline rejection; optimizer alignment recovery with an outlier.
- `pipelines/fusion/tests/gnss_interpolation.rs`: nearest/interpolated
  selection, covariance blending and motion inflation, dropouts, covariance
  precedence.
- `pipelines/slam/tests/gnss_fusion.rs`: end-to-end synthetic drives — fused
  ATE well below VO-only, every outlier-touched frame rejected and none
  elsewhere, outliers do not pull the trajectory (vs plain least squares),
  dropout bridged, monocular scale recovered, sliding window,
  unobservable-alignment / bad-timestamp errors, and a 10 Hz camera / 1 Hz
  receiver drive where only `gate_at_fix_accuracy` flags a 12 m jump, and
  one corrupted VO step that `vo_robust` absorbs.
- `crates/io/src/kitti_raw_gnss.rs` unit tests: WGS84/ECEF reference values,
  ENU axes, the OXTS roll/pitch/yaw convention and meridian convergence, GNSS
  fix conversion, and the raw calibration parser and IMU → rectified-camera
  chain.

## Limits

- Position-only GNSS: no velocity, heading, or raw-observation factors, and no
  receiver clock or multipath modelling beyond robust down-weighting.
- GNSS noise is treated as white per fix. Real receivers have time-correlated
  errors; the convex covariance blending and information sharing are
  conservative heuristics, not a colored-noise model. On KITTI raw with
  simulated 45 s-correlated errors this caps the gain over the raw fixes
  (seed means 6.0–6.6 m → 3.4–4.3 m, not the synthetic 5×).
- Robustness covers GNSS only by default: a VO failure (dynamic object,
  frontend lock-in) is trusted, and correct fixes near it are rejected
  instead. The opt-in `vo_robust` kernel handles a short failure such as
  0027's truck crossing. A VO that is wrong for hundreds of frames, such as
  the rescue lock-in, is too consistent for a kernel to reject.
- The VO noise model is a hand-set per-step sigma; VO covariances from the
  front end are not consumed yet, and the VO factors are only between
  consecutive frames (no loop closures inside this graph).
- Yaw-only alignment assumes a gravity-aligned map with `+z` up. With a full
  rotation, roll about a straight-line trajectory is unobservable until the
  path turns.
- The ENU frame is whatever the fixes are expressed in. Geodetic (WGS84) to
  local ENU conversion is in `visloc_io::kitti_raw_gnss::LocalEnuFrame`; the
  fusion itself does not convert.
- The sliding window has no marginalization prior on the frames it drops; the
  oldest window frame is simply held fixed.
- Real-data validation is three KITTI raw drives with the OXTS GPS/INS as
  ground truth and a **simulated** degraded receiver derived from it; no run
  with a real consumer receiver's output is claimed.
