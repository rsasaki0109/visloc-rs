# Faithful port of COLMAP's rig-aware incremental SfM into visloc-rs

Scope: `benchmarks/electro/m9-openloris-colmap-graph-isolation-10000-v1.json`
("m9-graph-isolation") shows that when visloc's `generalized_rig_sfm` mapper is
fed COLMAP's *exact* verified graph (same 70000 candidate pairs, same rig
config), it reproduces COLMAP's component structure almost exactly
(`registered_frames: 4493` vs COLMAP's `4494`) but collapses in scale
(`sim3_scale: 0.1356`, `ate.rmse_m: 11.88`, `rpe.10s.rmse_m: 8.23` vs COLMAP's
`0.305`/well under 1m), and the remaining 507-frame island fails outright
(`"stop": "no frame has enough multi-sensor tracks for metric initialization"`,
`registered_frames: 0`). COLMAP achieves this with only **111** same-frame
cross-sensor (stereo) verified pairs in the whole 10k-image graph
(`candidate_verification_comparison.by_class."cross-sensor gap=0".both = 111`,
identical between visloc and COLMAP — this is not a matching/frontend gap).
This document is a from-source, cited plan to port COLMAP's rig-aware
incremental mapper "as is" (ベタ移植) into visloc-rs, validated against COLMAP
on the same database, before any performance/memory deviation.


## Lead review (2026-09-14) — supersedes §5 where they conflict

- Reference values: the COLMAP 10k control components have Sim(3) scales
  1.106 (8296 scored images) and 1.142 (1010), ATE 0.3843 m, RPE-10s 0.305 m
  (coverage 0.931).
- §5 as written patches visloc behaviours step by step (Path A, gating,
  global BA, GP3P). The user asked for a faithful port first, so the new
  `colmap_incremental` module must follow COLMAP's `IncrementalPipeline` /
  `IncrementalMapper` control flow end to end, reusing visloc solvers only
  where they are numerically equivalent (P3P, fixed-rig reprojection
  residual, LM BA with trivial loss). visloc-specific heuristics
  (`min_pnp_sensors`, metric-anchored tracks, metric seeds, bounded BA) are
  not carried into the module.
- Revised milestones:
  - **C0** baseline: run today's `generalized_rig_sfm` (C′ flags) on the
    existing COLMAP 1k control database (`corridor1-1-m8-colmap/tier-1000-rig-v3`)
    via the isolation harness; record registered frames, components, Sim(3)
    scale, ATE, RPE next to COLMAP's 1k model. No code changes.
  - **C1** data model and inputs: `DatabaseCache` from the COLMAP export
    (images, cameras, rig/frames, keypoints, verified two-view geometries,
    correspondence graph), `Reconstruction`/`Rig`/`Frame` with
    `Frame::SetCamFromWorld`; unit tests against COLMAP semantics.
  - **C2** incremental loop with COLMAP option defaults of the control:
    initial pair selection, `RegisterNextImage` Path A (single camera P3P +
    rig composition), triangulation, local BA, retriangulation, filtering,
    ratio-triggered global BA, multiple models. Acceptance on 1k: registered
    frame set within ±1% of COLMAP, Sim(3) scale within 10% of COLMAP's,
    ATE within 1.5× of COLMAP's 1k value.
  - **C3** `RegisterNextGeneralFrame` (Path B) with a GP3P port; 2.5k/10k
    parity: components match 4494/505 within ±1%, ATE ≤ 1.2× COLMAP.
  - **C4** held-out sequence (corridor1-2) parity, then speed/memory work.

---

## 0. COLMAP version pinned for this port

- Control run: `benchmarks/electro/m8-openloris-colmap-10k-control.json` records
  `"colmap_version": "4.2.0.dev0"` and Docker image `colmap/colmap:latest`
  digest `sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`
  (also recorded in `.../tier-10000-rig-v3/plan.json:docker.id`). The binary
  itself reports `COLMAP 4.2.0.dev0 (Commit Unknown on Unknown ...)` — the
  Docker image does not embed a git SHA — but `docker image inspect
  colmap/colmap:latest --format '{{.Created}}'` gives `2026-07-29T08:19:29Z`.
- **Commit used for all citations in this document**:
  `64805cb870b574a569dccc34918d95a2db2b2fee` — the last commit on
  `colmap/colmap@main` at or before that image-build timestamp (`git log
  --until=2026-07-29T08:19:29Z`, author date `2026-07-28T17:35:04Z`,
  "Mitigate vcpkg app-local Windows CI race (#4596)"). This is the closest
  reproducible proxy for the exact source the control binary was built from;
  it should be re-verified against the *actual* `colmap/colmap:latest` digest
  if this plan is executed later (image "latest" moves).
- Source was fetched read-only via
  `https://raw.githubusercontent.com/colmap/colmap/<commit>/<path>` into
  `/tmp/.../scratchpad/colmap-port/colmap-src/` (path separators replaced with
  `_`), not built. All file:line citations below are against this commit.

### 0.1 Control mapper options (from `tier-10000-rig-v3/plan.json:commands` and `logs/`)

```
feature_extractor: --SiftExtraction.max_num_features 256 --SiftExtraction.first_octave -1
                    --SiftExtraction.peak_threshold 0.0066666667 --SiftExtraction.max_num_orientations 2
                    --FeatureExtraction.max_image_size 848 --ImageReader.single_camera_per_folder 1
rig_configurator:   --rig_config_path rig_config.json   (2 cameras, 1 rig, calibrated fixed stereo baseline)
mapper:             --Mapper.multiple_models 1 --Mapper.max_num_models 50 --Mapper.min_model_size 10
                    --Mapper.num_threads 8 --Mapper.abs_pose_min_num_inliers 8
                    --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0
                    --Mapper.ba_refine_extra_params 0 --Mapper.ba_use_gpu 0
                    --Mapper.random_seed 0 --Mapper.ba_refine_sensor_from_rig 0
```

The single most important flag for this investigation is
**`--Mapper.ba_refine_sensor_from_rig 0`**: the calibrated stereo baseline
(`sensor_from_rig`, the physically-measured extrinsic between the two
cameras) is **held fixed** through the entire incremental reconstruction and
every bundle adjustment. All other unset `IncrementalMapper::Options` /
`IncrementalPipelineOptions` fields are COLMAP defaults, cited in §1.2 below
(e.g. `init_min_num_inliers=100`, `ba_global_frames_ratio=1.1`,
`filter_max_reproj_error=4.0`, `min_track_length=0`, loss `TRIVIAL`/scale
`1.0` for BA by default). `mapper.log` confirms the calibrated init path
(`bundle_adjustment_ceres.cc:383` gauge-fixing warnings at the very first,
degenerate 2-image BA attempt, resolved as soon as ≥3 frames are registered).

---

## 1. COLMAP rig-aware incremental pipeline: source walkthrough

### 1.1 Data model — `Rig`, `Frame`, `sensor_from_rig`

- `src/colmap/sensor/rig.h:49-110` — class `Rig`: a rig is a set of sensors
  (`sensor_t`) with one designated **reference sensor** (identity pose in the
  rig frame, `Rig::AddRefSensor`, `rig.h:57`) and, for every other sensor, an
  optional but normally-known `sensor_from_rig: Rigid3d`
  (`Rig::SensorFromRig`, `rig.h:83-84`, backed by
  `sensors_from_rig_: std::map<sensor_t, std::optional<Rigid3d>>`,
  `rig.h:109`). For the OpenLORIS stereo rig this is exactly the calibrated
  camera-to-camera extrinsic (in **meters** — it is a physical measurement,
  not a free unit).
- `src/colmap/scene/frame.h:44-132` — class `Frame`: the atomic *registration
  unit*. A `Frame` owns a `rig_from_world_: std::optional<Rigid3d>` (the one
  6-DoF pose that is actually optimized) plus a `rig_ptr_` and a set of
  `data_ids_` (one per sensor image captured at that instant).
  `Frame::SensorFromWorld(sensor_id)` (`frame.h:210-217`) returns
  `RigFromWorld()` for the reference sensor and
  `rig_ptr_->SensorFromRig(sensor_id) * RigFromWorld()` otherwise — i.e. a
  non-reference camera's world pose is **always** the fixed baseline composed
  with the one shared rig pose, never an independently-optimized quantity.
- `src/colmap/scene/frame.cc:85-94` — `Frame::SetCamFromWorld(camera_id,
  cam_from_world)`: **this is the crux of COLMAP's metric guarantee.** If
  `camera_id` is the reference sensor, `RigFromWorld() = cam_from_world`
  directly. If it is *not* the reference sensor, it computes
  `RigFromWorld() = Inverse(cam_from_rig) * cam_from_world`, where
  `cam_from_rig` is the **fixed, metric** baseline. This means an *ordinary,
  single-camera* absolute-pose estimate for the second camera is converted
  into the frame's rig pose by composing with a rigid transform of known,
  non-scalable metric length — the composition cannot introduce or absorb a
  scale factor.

### 1.2 `IncrementalMapper::Options` / `IncrementalPipelineOptions` (control-relevant defaults)

`src/colmap/sfm/incremental_mapper.h:70-173`, `src/colmap/controllers/incremental_pipeline.h:47-215`:

| Field | Default (control uses default unless flagged in §0.1) |
|---|---|
| `init_min_num_inliers` | 100 |
| `init_max_error` | 4.0 px |
| `init_max_forward_motion` | 0.95 |
| `init_min_tri_angle` | 16° |
| `abs_pose_max_error` | 12.0 px |
| `abs_pose_min_num_inliers` | **8** (control override) |
| `abs_pose_min_inlier_ratio` | 0.25 |
| `ba_local_num_images` | 6 |
| `filter_max_reproj_error` | 4.0 px |
| `filter_min_tri_angle` | 1.5° |
| `max_reg_trials` | 3 |
| `multiple_models` / `max_num_models` / `min_model_size` | **1 / 50 / 10** (control) |
| `ba_refine_focal_length/principal_point/extra_params` | **0/0/0** (control: calibrated, fixed intrinsics) |
| `ba_refine_sensor_from_rig` | **0** (control: fixed baseline) |
| `ba_global_frames_ratio` / `ba_global_points_ratio` | 1.1 / 1.1 (re-run **global** BA every ~10% growth) |
| `ba_global_frames_freq` / `ba_global_points_freq` | 500 / 250000 |
| `random_seed` | **0** (control) |

### 1.3 Initial pair selection — `IncrementalMapperImpl::FindInitialImagePair`

`src/colmap/sfm/incremental_mapper_impl.cc:190-747` (entry at 190; two-view
geometry + gating at 680-747, quoted in full below):

- Ordinary calibrated/uncalibrated two-view geometry is estimated first
  (`EstimateTwoViewGeometry`, `EstimateTwoViewGeometryPose`,
  `impl.cc:688-694`) and gated on `init_min_num_inliers`,
  `init_max_forward_motion`, `init_min_tri_angle` (`impl.cc:703-709`) — this
  part is camera-pair-agnostic (frame membership is not consulted yet).
- **Only after** the pair passes ordinary two-view gating does rig-awareness
  enter (`impl.cc:711-737`): if either image's rig has `NumSensors() > 1`, the
  pair is re-solved with `EstimateInitialGeneralizedTwoViewGeometry`
  (a generalized/multi-camera relative-pose path — see §1.6); otherwise the
  plain `cam2_from_cam1` from the standard two-view solve is kept
  (`impl.cc:739`). There is **no special-case rejection of same-frame
  pairs** — COLMAP does not need to avoid picking two images of the very same
  rig instant as the seed; it only needs *some* valid two-view geometry
  passing the ordinary thresholds, from the full candidate-pair pool (of
  which the control run's 10k-image graph offers 111 same-frame cross-sensor
  pairs among 70000 candidates plus thousands of ordinary same-sensor
  cross-time pairs).
- `IncrementalMapper::FindInitialImagePair`
  (`src/colmap/sfm/incremental_mapper.cc:154-180`) wraps this and calls
  `SeedEstimatedInitialCameras` to seed intrinsics from the two-view solve.
  `RegisterInitialImagePair` (`incremental_mapper.cc:194-231`) then just calls
  `image.FramePtr()->SetCamFromWorld(...)` for each of the two images (§1.1's
  mechanism) and registers both frames.

### 1.4 `RegisterNextImage` — the dual registration paths, and why one camera is enough

`src/colmap/sfm/incremental_mapper.cc:233-490`. This is the single most
important function for the scale question. Key branch, `incremental_mapper.cc:250-271`:

```cpp
// Use central camera pose estimation for trivial frames and when we don't
// have a good estimate of the camera's focal length, because we don't have a
// focal length estimator for non-central/generalized cameras.
if (image.FramePtr()->RigPtr()->NumSensors() > 1) {
  bool all_cameras_have_good_focal_length = true;
  ...
  if (all_cameras_have_good_focal_length) {
    return RegisterNextGeneralFrame(options, *image.FramePtr());   // multi-camera generalized PnP
  }
}
// ... falls through to ordinary single-camera path below ...
```

- **Path A — ordinary single-camera path (the common case once intrinsics
  are known)**: `incremental_mapper.cc:273-489`. Gathers 2D-3D correspondences
  for **this one image only** (`correspondence_graph->FindCorrespondences`,
  `:301-336`), runs plain `EstimateAbsolutePose`/`RefineAbsolutePose`
  (ordinary P3P/EPnP-class solver, `:434-463`), then calls
  `image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world)`
  (`:472`) — which is exactly `frame.cc:85-94` from §1.1. **This is the
  proof that COLMAP does not need both cameras of a rig frame to have 2D-3D
  correspondences to register that frame; a single camera's monocular PnP
  is sufficient, and the frame's rig pose is derived from it via the fixed
  metric baseline.** Once the ordinary path has run for enough images that
  every rig camera has "a good estimate of focal length"
  (`num_reg_images_per_camera[...] > 0` and not bogus,
  `incremental_mapper.cc:254-266`), later frames route to Path B below, but
  Path A remains available (and is what the control run — fixed calibrated
  intrinsics, `ba_refine_focal_length=0` — is effectively always eligible
  for after the very first few images of each camera).
- **Path B — generalized (multi-camera) path**:
  `RegisterNextGeneralFrame`, `incremental_mapper.cc:492-669`. Pools 2D-3D
  correspondences from **all** images of the frame (`:516-575`, iterates
  `frame.ImageIds()`; a camera with zero correspondences simply contributes
  none — it is not a hard requirement that every sensor have hits), calls
  `EstimateGeneralizedAbsolutePose` (`:608-619`) then
  `RefineGeneralizedAbsolutePose` (`:631-641`), and sets
  `frame.SetRigFromWorld(rig_from_world)` directly (`:650`) — no central
  camera pose ever computed, no rig composition step needed because the
  solver itself is rig-aware (see §1.6).
- Either path calls `obs_manager_->RegisterFrame(...)` — **frames**, not
  images, are the registration unit and either path can independently
  succeed with correspondences from as few as one sensor.

### 1.5 Generalized pose solvers (`src/colmap/estimators/generalized_pose.cc`, `.../solvers/`)

- `EstimateGeneralizedAbsolutePose` (`generalized_pose.cc:131-190`) sets up
  `RANSAC<GP3PEstimator, UniqueInlierSupportMeasurer>`
  (`generalized_pose.cc:176-178`) — the minimal solver is **GP3P**
  (`estimators/solvers/generalized_absolute_pose.h/.cc`), a **3-correspondence**
  polynomial minimal solver generalized to allow the 3 rays to originate from
  different rigidly-related cameras (it degenerates cleanly to ordinary P3P
  when all 3 rays come from one camera). Because the sample size is 3, GP3P
  can and does solve using correspondences from a single camera when that is
  all that is available in a sample — it does **not** require ≥2 distinct
  cameras to be represented.
- `EstimateGeneralizedRelativePose` (`generalized_pose.cc:192-...`) uses
  `LORANSAC<GR6PEstimator, GR8PEstimator>` (`generalized_pose.cc:267`) — a
  6-point (minimal) / 8-point (non-minimal LO refinement) generalized
  relative-pose solver family, used by
  `EstimateInitialGeneralizedTwoViewGeometry` (§1.3) when either frame of the
  initial pair is a non-trivial rig.
- `RefineGeneralizedAbsolutePose` (`generalized_pose.cc:280-...`) is a small
  Ceres refinement over just the inlier set, holding `cams_from_rig` (the
  fixed baseline) constant and refining only `rig_from_world` — i.e. it
  reuses the same fixed-baseline formulation as full BA (§1.7), so **the
  pose returned to `RegisterNextGeneralFrame` is already exactly
  baseline-consistent before any bundle adjustment ever runs.**
- Net effect for scale: whichever path registers a frame (§1.4's Path A or
  Path B), the **fixed, metric `sensor_from_rig` is either a direct input to
  the solver (GP3P/refinement) or is applied as an exact rigid composition
  after an ordinary single-camera solve (`Frame::SetCamFromWorld`)** — in
  neither case is scale a free parameter of the registration step itself.

### 1.6 Bundle adjustment — `BundleAdjustmentConfig`, cost functors, gauge fixing

`src/colmap/estimators/bundle_adjustment.h/.cc`,
`src/colmap/estimators/bundle_adjustment_ceres.cc`,
`src/colmap/estimators/cost_functions/reprojection_error.h`.

- **Config surface** (`bundle_adjustment.h:77-150`): per-rig
  `SetConstantSensorFromRigPose(sensor_t)` / `SetVariableSensorFromRigPose`,
  per-frame `SetConstantRigFromWorldPose(frame_t)`, per-camera
  `SetConstantCamIntrinsics(camera_t)`, plus a `BundleAdjustmentGauge`
  (`UNSPECIFIED` / `TWO_CAMS_FROM_WORLD` / `THREE_POINTS`,
  `bundle_adjustment.h:47-48`).
- **Cost functors** (`cost_functions/reprojection_error.h`):
  - `ReprojErrorCostFunctor<CameraModel>` (`:217-253`): variable
    `(point3D, cam_from_world[7], camera_params)` — plain single-camera BA,
    used for `IsRefInFrame()` images with variable pose.
  - `ReprojErrorConstantPoseCostFunctor` (`:261-307`): variable
    `(point3D, camera_params)`, pose baked in as a **constructor argument**
    (precomputed rotation matrix + translation, not a Ceres parameter block
    at all) — used when the whole `cam_from_world` is fixed.
  - `RigReprojErrorCostFunctor<CameraModel>` (`:344-384`): variable
    `(point3D, cam_from_rig[7], rig_from_world[7], camera_params)` — full rig
    BA with the baseline as a free parameter block (only reached when
    `ba_refine_sensor_from_rig=1`, **not** the control configuration).
  - **`RigReprojErrorConstantRigCostFunctor<CameraModel>` (`:389-417`) — the
    control-configuration functor.** Constructor takes `cam_from_rig` **by
    value** and stores it as a plain member (`:397-399,406`); `operator()`'s
    template parameter list is only `(point3D, rig_from_world, camera_params)`
    — **the baseline is not a Ceres parameter block at all in this path; it
    is compiled into the residual as data.** This is invoked from
    `AddImageWithNonTrivialFrame` (`bundle_adjustment_ceres.cc:793-800`) whenever
    `constant_sensor_from_rig && !constant_rig_from_world` — exactly the
    control run's case (`ba_refine_sensor_from_rig=0`, frame poses variable).
  - When *both* `sensor_from_rig` and `rig_from_world` are constant
    (e.g. a fixed/anchor frame), COLMAP additionally **precomposes**
    `cam_from_world = sensor_from_rig * rig_from_world` once
    (`bundle_adjustment_ceres.cc:759-762`) and falls back to the plain
    `ReprojErrorConstantPoseCostFunctor` — an optimization, not a semantic
    change.
- **Why this enforces metric scale (not just "gauge-fixes" it)**: with the
  baseline hard-coded into every non-reference-camera residual for every
  rig frame in the connected optimization graph
  (`ParameterizeRigsAndFrames`, `bundle_adjustment_ceres.cc:463-536`, sets
  `problem.SetParameterBlockConstant(sensor_from_rig.params.data())` at
  `:491` whenever `!options.refine_sensor_from_rig`), a *global* rescaling
  of the reconstruction by any factor `s ≠ 1` would have to leave every
  camera-2 residual unchanged, but the fixed baseline used inside that
  residual does **not** scale with `s` — so residuals grow, the Hessian has
  no null direction along scale, and scale is a **fully observed, not a
  gauge, degree of freedom** wherever at least one rig-frame residual with a
  non-reference camera exists in the connected graph (this is a **different
  statement from Gauge-fixing**, see next point).
- **Gauge fixing is orthogonal to metric scale.** `FixGaugeWithTwoCamsFromWorld`
  (`bundle_adjustment_ceres.cc:300-409`) / `FixGaugeWithThreePoints` (`:262-293`)
  remove the *unobservable* 6-7 DoF (position + orientation, and — only in
  ordinary monocular BA without a fixed baseline anywhere — scale) that would
  otherwise leave the normal-equations singular; they anchor 1-2 frame poses
  (or 3 points) as `SetParameterBlockConstant` so Ceres has a unique solution,
  independent of *which* solution is metrically correct. The two mechanisms
  compose: gauge-fixing prevents numerical drift of the *arbitrary* global
  reference frame, while the fixed baseline (previous bullet) prevents scale
  from being arbitrary in the first place. The `mapper.log` warning
  (`"Failed to fix Gauge with two cameras. Falling back to fixing Gauge with
  three points."`, from `bundle_adjustment_ceres.cc:383-384`) fires only in
  the degenerate 2-image, ~1-point global BA right after a bad initial-pair
  attempt — it recurs a few times in `logs/mapper.log` and is unrelated to
  the steady-state scale question once ≥3 frames are registered.
- **Loss / solver**: default loss is `TRIVIAL` (plain L2, no robust kernel)
  at scale 1.0 (`bundle_adjustment_ceres.h:43-47`); linear solver is
  auto-selected by problem size — `DENSE_SCHUR` ≤ 50 images,
  `SPARSE_SCHUR` ≤ 1000, else `ITERATIVE_SCHUR` + `SCHUR_JACOBI`
  preconditioner (`bundle_adjustment_ceres.cc:194-205`); trust-region
  strategy is Ceres' default (Levenberg-Marquardt) — nothing in this file
  overrides `trust_region_strategy_type`.

### 1.7 Global vs. local BA cadence, filtering, model splitting

`src/colmap/controllers/incremental_pipeline.cc` (878 lines; `Reconstruct`,
`ReconstructSubModel`, `InitializeReconstruction`,
`CheckRunGlobalRefinement`) and `IncrementalMapper::AdjustLocalBundle` /
`AdjustGlobalBundle` (`incremental_mapper.cc`, `~1000-1250`):

- **Global BA re-runs the *entire* connected reconstruction** every time
  registered frames or points grow by the configured ratio
  (`ba_global_frames_ratio=1.1`, `ba_global_points_ratio=1.1`,
  §1.2) or every `ba_global_frames_freq=500`/`ba_global_points_freq=250000`
  frames/points, whichever comes first (`CheckRunGlobalRefinement`). At 10k
  images / ~5000 frames this means global BA over the **whole graph** fires
  roughly every ~450-500 new frames. Because §1.6's fixed-baseline residual
  sits inside this single joint Ceres problem together with every other
  frame and point, **the metric constraint from even one rig-frame
  observation anywhere in the connected component is propagated to every
  other frame and point through the shared normal equations at every one of
  these global passes** — this is the load-bearing mechanism that lets 111
  same-frame stereo pairs anchor a 4494-frame reconstruction.
- Local BA (`AdjustLocalBundle`, windowed to `ba_local_num_images=6`
  around the most recently registered frame) runs after every single frame
  registration and is a genuine local/windowed optimization — but it does
  not need to carry scale itself; it only needs to not actively destroy it,
  and it uses the *same* constant-baseline cost functors as global BA
  (§1.6), so any camera-2 observation inside a 6-image window still pins
  that window's scale exactly, and even a purely-monocular window inherits
  correct scale from its (already metric) fixed/boundary poses.
- Model splitting (`multiple_models`/`max_num_models`/`min_model_size`,
  the control sets `1`/`50`/`10`) creates a **new independent sub-reconstruction**
  (its own `IncrementalMapper::BeginReconstruction`) once the current model's
  frontier is exhausted — this is exactly why the control run produced 2
  models (`registered_frames: 4494` and `505`, `result.json`): the second
  model is bootstrapped completely independently, with its **own** initial
  pair search (§1.3) and its own from-scratch metric anchor — it does not
  inherit scale from model 0. COLMAP does not need "enough" stereo pairs in
  that sub-graph either — by §1.3/§1.4, exactly one successful rig-baseline
  event (an initial pair using `EstimateInitialGeneralizedTwoViewGeometry`,
  or the first Path-A/Path-B frame registration that touches a non-reference
  camera) is sufficient to lock that sub-reconstruction's scale for good.
- `FilterFrames`/`FilterPoints` (`incremental_mapper.cc:1319-1385`) apply the
  ordinary `filter_max_reproj_error`/`filter_min_tri_angle` thresholds
  per-observation; nothing rig-specific beyond using
  `Frame::SensorFromWorld` (§1.1) to compute per-image reprojection.

---

## 2. visloc-rs counterpart mapping and root-cause of the scale collapse

Primary files: `pipelines/slam/src/rig_sfm.rs` (10321 lines, the rig mapper
itself), `pipelines/slam/src/bundle.rs` (16346 lines, rig-aware BA),
`pipelines/slam/src/camera_rig.rs` (569 lines, per-image camera / rig
container used by the *non*-rig incremental path),
`crates/vision/src/pnp/generalized.rs` (`GeneralizedPnPRansac`,
`GeneralizedDltPoseEstimator`), `crates/vision/src/pnp/gr6p.rs` (generalized
relative pose), `examples/generalized_rig_sfm.rs` (4492-line CLI harness used
by the M9 benchmark). `pipelines/slam/src/incremental_sfm.rs` is the
**non-rig, monocular** COLMAP-schedule port (`docs/colmap_port_plan.md`'s
subject, §2 table) — it is a different mapper from `rig_sfm.rs` and is not
directly on the critical path for this gap, though its `IncrementalMapper`
schedule (visibility-pyramid next-best-view, local+global BA cadence,
retriangulation, filtering) is architecturally the model to imitate at the
scheduling layer (§4).

| COLMAP component | visloc-rs counterpart | Verdict |
|---|---|---|
| `Rig` / `Frame` / `sensor_from_rig` (§1.1) | `GeneralizedCameraRig`/`RigSensor` (`crates/vision/src/pnp/generalized.rs:30-38`), `RigFrame`/`RigFrameImage` (`rig_sfm.rs:37-46`) | **Identical in shape** — a frame is a set of images with one shared pose and per-sensor fixed `sensor_from_rig: SE3`. |
| `Frame::SetCamFromWorld` composing a monocular pose through the fixed baseline (§1.1, §1.4 Path A) | **No equivalent exists.** `rig_sfm.rs` has no code path that registers a frame's `RigFromWorld` from a single camera's independent monocular PnP composed with `sensor_from_rig`. | **Missing.** This is architecturally central — see root cause below. |
| `IncrementalMapper::RegisterNextImage` dual path (central-camera fallback + generalized) (§1.4) | `incremental_rig_sfm` registration loop (`rig_sfm.rs:857-...`) always calls the **generalized** solver (`GeneralizedPnPRansac::estimate`, e.g. `rig_sfm.rs:1145,2780,6556,8752`) | **Deviates.** visloc has only COLMAP's "Path B"; COLMAP's "Path A" (ordinary single-camera PnP + `SetCamFromWorld`) has no counterpart. |
| `min_pnp_sensors` requirement for a registration attempt | `RigSfmConfig::min_pnp_sensors`, **default `2`** (`rig_sfm.rs:287`), enforced at `rig_sfm.rs:1414` (`if sensors < config.min_pnp_sensors` → frame not attempted) | **Deviates, and is the single largest contributor to the gap.** COLMAP's effective minimum is 1 camera (Path A). visloc by default refuses to even attempt registering a frame unless **both** rig cameras have usable 2D-3D correspondences at that exact instant. |
| `GP3PEstimator`, minimal sample = 3 (§1.5) | `GeneralizedDltPoseEstimator`, a **linear DLT**, `MINIMUM_CORRESPONDENCES = 6` (`crates/vision/src/pnp/generalized.rs:151`) | **Deviates.** visloc's generalized solver needs double COLMAP's minimal sample (6 vs 3), and — being a linear DLT rather than a calibrated polynomial minimal solver — does not exploit the known intrinsics/rig geometry as tightly; this materially raises the correspondence bar for every registration attempt, not just the seed. |
| Per-track metric trust: **none** — any track is trusted once its parent frame poses are registered (§1.1, §1.6) | `track_is_metric_anchored` (`rig_sfm.rs:3813-3821`): a track is "metric anchored" only if **that specific track** has ≥2 observations from different sensors in the same frame; `retain_metric_anchored_tracks` (`rig_sfm.rs:3801-3811`) discards non-anchored tracks entirely in some track-builder policies; `RigSfmConfig::ba_metric_tracks_only` can further restrict BA to only such tracks. | **Deviates — a structural, not incidental, difference.** COLMAP's metric guarantee is a property of the **pose graph** (the fixed baseline inside every non-reference-camera residual, §1.6); visloc instead tries to certify metric-ness **per 3-D point**, which is strictly harder to satisfy at scale and is unnecessary given a correct pose-graph-level guarantee. |
| Metric seeding: any one already-registered frame anywhere in the connected component (transitively, via monocular PnP chains) suffices to lock scale for the *whole* component (§1.4, §1.7) | `metric_frame_supports`/`metric_seed_candidates` (`rig_sfm.rs:5735-5774`): a **seed frame** must have **≥6** metric-anchored (cross-sensor) tracks (`supported >= 6`, `rig_sfm.rs:5769`, matching the DLT's `MINIMUM_CORRESPONDENCES=6`); if no frame in the (sub)graph clears this bar, seeding fails outright with `RigSfmError::NoMetricSeed` (`rig_sfm.rs:559`, message `"no frame has enough multi-sensor tracks for metric initialization"` — the exact string in the M9 result for the 507-frame island). | **Deviates, and directly explains the M9 island failure.** This is a strictly *local*, per-frame bootstrap requirement, whereas COLMAP's seeding requirement is essentially "≥1 valid two-view pair anywhere" (§1.3) plus, separately, "≥1 rig-baseline event anywhere in the component, ever" (§1.4) — decoupled from where or how many. |
| Rig-fixed reprojection cost functor `RigReprojErrorConstantRigCostFunctor` (§1.6) | `BaRigObservation { sensor_from_rig: SE3, ... }` (`bundle.rs:981-990`), consumed in `add_rig_observation` (`bundle.rs:1581`) and the residual at `bundle.rs:9399-9409` — `sensor_from_rig` is stored per-observation as a plain field (constant), matching COLMAP's baked-in-constant formulation. | **Identical in formulation** — this part of the BA is already a faithful port. The residual math is not the source of the scale collapse. |
| Global BA re-run over the **whole connected graph** every ~10% growth (§1.7) | `run_rig_bundle_adjustment` (`rig_sfm.rs:4906-...`) takes an explicit `active_frames: &HashSet<usize>` **window**; `RigBaBackend` variants are explicitly bounded/windowed (`MatrixFreeCluster8` "at most eight poses", `MatrixFreeQr`, `BoundedDirect64Qr` switches to windowed QR above 64 variable poses, `rig_sfm.rs:96-117`). No code path found that performs a COLMAP-style full-graph joint BA over thousands of frames as a matter of course. | **Deviates, and is the second contributor to the 0.136 scale collapse.** Even where a frame *does* pass the `min_pnp_sensors=2`/`≥6`-track bars once, COLMAP's design keeps re-tying that frame's scale to the global metric anchor via full-graph BA every ~500 frames; a windowed/bounded BA whose window boundaries frequently contain **zero** metric-anchored observations (111 stereo pairs spread across ~4500 frames means most 6-64-frame windows see none) has no mechanism to detect or correct accumulated scale drift the way a single joint Hessian does. This is consistent with the *shape* of the failure (a nearly-uniform ~0.136 global shrink plus large per-segment ATE, not a local artifact). |
| Gauge fixing (`TWO_CAMS_FROM_WORLD`/`THREE_POINTS`, §1.6) | `problem.fix_pose(anchor_frame_index)` (`rig_sfm.rs:4931`), `rig_ba_component_anchors` (`rig_sfm.rs:4873`) | **Equivalent in spirit** (anchor one frame's pose per connected component) — not implicated in the scale bug, since gauge-fixing and metric-scale are orthogonal (§1.6). |

### 2.1 Root cause, stated precisely

The M9 evidence (`candidate_verification_comparison.by_class."cross-sensor
gap=0"`) proves visloc and COLMAP see **exactly the same 111 same-frame
stereo pairs** — this is not a frontend/matching gap. Given that, three
independent, additive, and all-cited design choices in `rig_sfm.rs` explain
the failure:

1. **Frame-registration gate (`min_pnp_sensors=2`, `rig_sfm.rs:287,1414`)**
   refuses to attempt any frame unless *both* cameras have usable
   correspondences *at that exact instant*, where COLMAP's `RegisterNextImage`
   (`incremental_mapper.cc:250-271`) will register a frame from **one**
   camera's monocular PnP and derive the other via `Frame::SetCamFromWorld`'s
   exact rigid composition (`frame.cc:85-94`). This alone means visloc drops
   every stretch of the sequence where good stereo overlap isn't
   simultaneously available on both sensors — plausibly the entire 507-frame
   island, and gaps inside the 4493-frame component that get bridged only by
   weaker, unconstrained tracks.
2. **Per-track metric gating** (`track_is_metric_anchored`,
   `rig_sfm.rs:3813-3821`, `ba_metric_tracks_only`) tries to re-derive, per
   3-D point, a guarantee that COLMAP gets for free at the pose-graph level
   (§1.6's Hessian argument) — this is not wrong, but it is a much stronger
   (and more failure-prone, given #1's already-scarce stereo evidence)
   requirement than necessary once frame poses are correctly propagated
   through the fixed baseline.
3. **Windowed/bounded BA as the default execution model**
   (`RigBaBackend::Legacy`'s `active_frames` window, and the
   `MatrixFreeCluster8`/`MatrixFreeQr`/`BoundedDirect64Qr` backends,
   `rig_sfm.rs:89-117,4906`) means the single mechanism that lets COLMAP's
   111 stereo pairs anchor 4494 frames — a full-graph joint BA run every
   ~500 frames (§1.7) — has **no direct counterpart**. Even where frames do
   pass gate #1, nothing periodically re-couples a distant frame's scale to
   the nearest metric anchor through one shared normal-equations solve.

None of this implicates the BA **cost function** itself (`BaRigObservation`,
`bundle.rs:981-990,9399`), which already faithfully bakes in the fixed
`sensor_from_rig` exactly as COLMAP's `RigReprojErrorConstantRigCostFunctor`
does (§1.6) — the bug is in the surrounding **mapper architecture**
(registration eligibility, track admission, and BA scheduling/scope), not in
the residual math. This matches `docs/colmap_port_plan.md`'s general finding
for the (unrelated) monocular gap: visloc's math/estimators are frequently at
parity or better than COLMAP's; the gaps are almost always in the
*scheduling and graph-structural* layers COLMAP wraps around them.

---

## 3. Port design

### 3.1 New module layout

Add a new module tree that mirrors COLMAP's naming exactly, so every future
diff against upstream COLMAP is a structural diff, not a re-derivation:

```
pipelines/slam/src/colmap_incremental/
  mod.rs                  // IncrementalPipeline equivalent (top-level Reconstruct loop)
  reconstruction.rs        // Reconstruction, Rig, Frame containers (mirrors scene/{reconstruction,rig,frame})
  database_cache.rs         // DatabaseCache (mirrors scene/database_cache) — reuse M2's CorrespondenceGraph if it exists
  mapper.rs                 // IncrementalMapper: FindInitialImagePair, RegisterNextImage (dual-path!), RegisterNextGeneralFrame
  mapper_impl.rs             // IncrementalMapperImpl: FindInitialImagePair internals, FindNextImages
  bundle_adjustment.rs        // BundleAdjustmentConfig + Ceres-equivalent cost functors (reuse pipelines/slam/src/bundle.rs's solver core)
  generalized_pose.rs          // EstimateGeneralizedAbsolutePose (GP3P) / EstimateGeneralizedRelativePose (GR6P/GR8P)
```

This is deliberately **separate from `rig_sfm.rs`**, not a patch to it:
`rig_sfm.rs`'s track-builder/metric-anchoring machinery
(`RigTrackBuilder::MetricAnchoredCycle` etc.) is a different, more
conservative design point that other callers may depend on; the faithful
COLMAP port should be validatable in isolation (§4) before any decision is
made about deprecating or merging the two.

### 3.2 Data structures

- `Rig`/`Frame`/`sensor_t`/`frame_t`/`data_t` should be ported close to
  1:1 from `sensor/rig.h` (`rig.h:49-110`) and `scene/frame.h` (`frame.h:44-132`)
  — visloc's existing `GeneralizedCameraRig`/`RigSensor`
  (`crates/vision/src/pnp/generalized.rs:30-38`) and `RigFrame`/`RigFrameImage`
  (`rig_sfm.rs:37-46`) are already structurally equivalent and can likely be
  reused directly or with a thin adapter, rather than re-invented — the gap
  is behavioral (§2), not representational.
- **Port `Frame::SetCamFromWorld` verbatim** (`frame.cc:85-94`) as the first
  concrete deliverable (§5, Milestone 1) — this single ~10-line function is
  what's structurally missing today (§2 row 2) and is the cheapest, most
  isolated way to add COLMAP's "Path A" registration.
- `CorrespondenceGraph`: if M2 of `docs/colmap_port_plan.md` (persistent
  correspondence graph, `docs/colmap_port_plan.md:225`) has landed by the
  time this work starts, reuse it verbatim — COLMAP's own
  `FindCorrespondences` (`incremental_mapper.cc:301-303`) is exactly what
  that milestone was scoped to provide. If not landed, this port should not
  block on it; a minimal read-only correspondence index scoped to this
  module is acceptable and should be flagged for future consolidation.

### 3.3 Solver reuse decisions

- **Absolute pose (Path A, single camera)**: visloc already has Grunert P3P
  (`crates/vision/src/pnp/p3p.rs`, per `docs/colmap_port_plan.md`'s verdict
  table, "EXISTS") — reuse directly for the ordinary
  `EstimateAbsolutePose`/`RefineAbsolutePose` COLMAP calls at
  `incremental_mapper.cc:434-463`.
- **Generalized absolute pose (Path B)**: COLMAP's GP3P
  (`estimators/solvers/generalized_absolute_pose.cc`, 3-point minimal,
  §1.5) has **no existing visloc equivalent** — `GeneralizedDltPoseEstimator`
  (6-point linear DLT) is a different algorithm with a materially worse
  minimal-sample requirement (§2 row 4). Porting an actual GP3P polynomial
  solver is recommended (not merely reusing the DLT) since it directly
  relaxes both the per-attempt correspondence floor and, transitively, the
  `metric_seed_candidates` `>=6` threshold (which is driven by the DLT's
  `MINIMUM_CORRESPONDENCES=6`, `generalized.rs:151` — a 3-point GP3P would
  let that threshold drop to COLMAP's implicit "≥1 rig-baseline event"
  requirement).
- **Generalized relative pose (initial pair)**: visloc already has
  `crates/vision/src/pnp/gr6p.rs` (a GR6P-family solver, per the
  `/tmp/colmap_gr6p*`/`/tmp/colmap_gr8p*` build artifacts found on disk from
  earlier exploratory work) — this appears to already be a faithful
  counterpart to COLMAP's `GR6PEstimator`/`GR8PEstimator`
  (`generalized_pose.cc:267`) and should be reused, not re-ported.
- **Bundle adjustment**: `pipelines/slam/src/bundle.rs`'s existing
  `BundleAdjustment`/`BaRigObservation` machinery (§2 row: "identical in
  formulation") is the correct base to build on — it already implements
  COLMAP's fixed-baseline residual faithfully. What needs to change is *how
  it is invoked* (§3.4), not its cost function:
  - Add a **full-graph BA mode** (all registered frames as `active_frames`,
    no windowing) alongside the existing bounded/windowed backends, gated
    by COLMAP's exact triggers (`ba_global_frames_ratio`/`points_ratio`/
    `freq`, §1.2) — this directly targets root cause #3 (§2.1).
  - Loss function: COLMAP's BA default is `TRIVIAL` (plain L2, no robust
    kernel, `bundle_adjustment_ceres.h:44`) for the control's calibrated
    path — check `pipelines/slam/src/bundle.rs`'s `RobustKernel` default
    against this; if visloc defaults to a robust kernel (e.g. Huber) where
    COLMAP does not, that is a numerical-parity deviation to document/switch
    off for the parity harness (§4), even though it is unlikely to explain
    the scale collapse on its own.
  - Solver: COLMAP uses Ceres' Levenberg-Marquardt with a Schur-complement
    linear solver (`DENSE_SCHUR`/`SPARSE_SCHUR`/`ITERATIVE_SCHUR` by size,
    §1.6). visloc's LM-family solver in `bundle.rs`
    (`LinearSolver`/`MatrixFreeBaOptions`) is the natural counterpart;
    exact numerical parity is not required (§6 risk), but the trust-region
    damping strategy and convergence criteria should be documented for the
    parity harness's tolerance-setting.

### 3.4 Registration/track-admission changes (the behavioral core of the port)

1. **Implement Path A** (§1.4): when a frame's non-seed camera lacks
   correspondences but the reference (or any single) camera has enough for
   ordinary `abs_pose_min_num_inliers`, register via single-camera PnP +
   `SetCamFromWorld`-style composition, exactly mirroring
   `incremental_mapper.cc:250-489`.
2. **Drop (or make optional, default off) the per-track
   `track_is_metric_anchored` gate** for triangulation/BA admission once
   Path A exists — COLMAP does not need it because pose-graph correctness
   is enough (§1.6, §2.1). Keep it available as a config flag for A/B
   comparison against the existing `rig_sfm.rs` design, but the *faithful
   port* module (§3.1) should not require it by default.
3. **Replace the `>=6`-per-frame `metric_seed_candidates` bar** with
   COLMAP's actual two-stage criterion: (a) find any valid two-view pair by
   ordinary thresholds (§1.3, independent of frame membership), (b) only
   invoke the generalized/rig-aware relative-pose solver if either image's
   rig is non-trivial. This requires the GP3P/GR6P solver work in §3.3 to
   avoid re-introducing a 6-point floor by another name.
4. **Add full-graph, ratio-triggered global BA** (§3.3) as the periodic
   scale-reconciliation step, matching `ba_global_frames_ratio=1.1` /
   `ba_global_frames_freq=500`.

### 3.5 Determinism strategy

- COLMAP's control run pins `--Mapper.random_seed 0`; RANSAC / candidate
  shuffling inside `IncrementalMapperImpl` and the GP3P/P3P RANSAC loops must
  accept an equivalent seed and be reproducible run-to-run — visloc's
  existing `RigSfmConfig::ransac_seed` (`rig_sfm.rs:132`) and
  `GeneralizedPnPRansac::seed` (`generalized.rs:370`) patterns already do
  this; carry the same discipline into the new module (one top-level seed,
  deterministically sub-derived per RANSAC call site, never wall-clock or
  thread-count dependent).
- Multi-threading: COLMAP's BA solver thread count affects only wall time,
  not the solved values (Ceres' Schur solve is deterministic given a fixed
  problem and initial values); the port should audit that any
  parallel/rayon reductions in the new module (track building, candidate
  correspondence gathering) are order-independent (e.g. sorted before
  reduction) so results are bit-reproducible across `--num-threads`, exactly
  as `docs/colmap_port_plan.md`'s existing M-series milestones require of
  their own debug instrumentation (`docs/colmap_port_plan.md:1736-1737`:
  "verified byte-identical registered counts/RMSE with and without the env
  var").

---

## 4. Parity test harness

### 4.1 Inputs

- Primary input: **COLMAP's own `database.db`** after `feature_extractor` +
  `rig_configurator` + `matches_importer`/exhaustive matching — i.e. the
  *verified* graph plus keypoints, exactly as
  `scripts/export_colmap_verified_for_visloc.py` /
  `examples/import_colmap_verified_snapshot` already consume for the M9
  benchmark (`benchmarks/electro/m9-openloris-colmap-graph-isolation-10000-v1.json`'s
  methodology). This removes the frontend/matching axis entirely and isolates
  the mapper, matching this document's own scope.
- Small tiers first, reusing **existing** COLMAP runs already on disk before
  running anything new:
  - `/home/sasaki/datasets/openloris/corridor1-1-m8-colmap/tier-1000-rig-v3/`
    and `tier-1000-same-candidate/` (both present, `result.json`/`models/`
    populated) — 1000-image tier, already has a COLMAP reference
    reconstruction to diff against.
  - `benchmarks/electro/m8-openloris-colmap-1k-control.json` records the 1k
    control's COLMAP numbers for direct comparison.
  - Only fall back to running COLMAP fresh (`colmap/colmap:latest` docker
    image, confirmed present locally: `docker images | grep colmap` →
    `colmap/colmap:latest b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`)
    via `scripts/benchmark_openloris_colmap.py` if a tier not already on disk
    is needed (e.g. a 2.5k intermediate tier).

### 4.2 Comparisons, in increasing order of strictness

1. **Initial pair**: exact `(image_id1, image_id2)` match against
   `logs/mapper.log`'s `"Registering initial image pair #X and #Y"` lines.
   (Tolerance: exact match not required in general — COLMAP's own selection
   has ties broken by iteration order over a heap of candidate pairs — but
   for a **fixed** `random_seed=0` and identical candidate pair set, exact
   match is the target; log a mismatch as a warning, not a hard failure,
   until milestone 3+.)
2. **Registration order**: sequence of frame ids registered, compared
   position-by-position with a Kendall-tau / longest-common-subsequence
   score rather than requiring byte-identical order (COLMAP's own
   `ImageSelectionMethod::MIN_UNCERTAINTY` visibility scoring has legitimate
   floating-point tie-breaks) — target ≥0.9 LCS ratio by milestone 3.
3. **Per-step registered frame ids**: set-equality of "ever registered"
   frame ids at the end (this is what M9 already measures: `4493` vs
   COLMAP's `4494`) — target: exact match (±0 frames) by milestone 3, since
   this is the headline reproduced-component-structure metric already
   achieved once (without correct scale) in M9.
4. **Points count / mean reprojection error**: compare `result.json`-style
   summaries (`points`, `observations`,
   `mean_reprojection_px_observation_weighted`) — target within COLMAP's own
   run-to-run noise band (establish this band by running the control twice
   if not already known; `random_seed=0` should make COLMAP itself
   deterministic, so the band may be ~0).
5. **Final Sim(3) scale and ATE/RPE vs COLMAP** (the headline metric this
   whole effort targets): reuse the exact per-component Sim(3)-alignment
   convention already used by `benchmarks/electro/m8-*`/`m9-*`
   (`"under the same per-component Sim(3) convention"`,
   `m8-openloris-colmap-10k-control.json:trajectory_target`). Target,
   staged by milestone (§5): first "scale within 2x of 1.0" (vs today's
   `0.1356`), then "ATE RMSE within 2x of COLMAP's `0.305`–`0.384` m band",
   then full parity (ATE RMSE ≤ COLMAP's own p95 band).

### 4.3 Harness implementation note

Extend `benchmarks/electro/m9-openloris-colmap-graph-isolation-10000-v1.json`'s
own methodology (feed COLMAP's *exact* verified graph to the visloc mapper,
already implemented per that benchmark's `pre_registration.mapper` field) as
the harness's backbone, rather than building a new one — it already produces
the `candidate_verification_comparison`, `ate`, and `mapper_ranks` sections
this plan's acceptance criteria are defined against. Add the new module's
mapper as an alternative `pre_registration.mapper` value alongside
`generalized_rig_sfm`, so the two can be A/B'd on identical inputs.

---

## 5. Milestones

| # | Milestone | Scope | Acceptance | Size |
|---|---|---|---|---|
| **P1** | **Port `Frame::SetCamFromWorld` + Path A single-camera registration fallback** into a new, minimal module (§3.1/§3.4 item 1), reusing existing P3P (`crates/vision/src/pnp/p3p.rs`) and the existing `BaRigObservation` cost function unmodified. Do **not** yet touch track admission or BA windowing. | On the 1k tier (`tier-1000-rig-v3`), run the new mapper against COLMAP's exact verified graph. | Registers at least as many frames as today's `generalized_rig_sfm` on the same tier, **and** at least one frame is registered via the new single-camera path where the old `min_pnp_sensors=2` gate would have rejected it (instrumented counter). No scale/ATE target yet — this milestone is "does Path A fire and not crash." | ~3-5 days (one function port + one new registration branch + counters) |
| **P2** | **Relax per-track/per-seed metric gating** (§3.4 items 2-3) behind a config flag defaulting to COLMAP's behavior (no per-track gate; seed = any valid two-view pair). Keep the existing `rig_sfm.rs` gated path available for regression A/B. | 1k tier, COLMAP verified graph. | `NoMetricSeed`/"no frame has enough multi-sensor tracks" failures on any component present in the 1k tier's COLMAP output drop to zero (i.e. every COLMAP-registered component also gets a visloc seed). Sim(3) scale moves from whatever P1 measured to within **2x of 1.0** (interim target, not final parity). | ~1 week (mostly plumbing + the parity harness's §4.2 items 1-3) |
| **P3** | **Full-graph global BA mode**, ratio/freq-triggered exactly per §1.2/§3.3, alongside existing windowed backends (kept for perf-sensitive callers). | 1k and 2.5k tiers. | Sim(3) scale within **2x of 1.0** *and* ATE RMSE within **2x of COLMAP's control band** (§4.2 item 5, stage 2) on both tiers. Registered-frame-id set matches COLMAP within ±1%. | ~1-2 weeks (BA scheduling + full-graph problem assembly at 2.5k-image scale; watch wall-time, not a target metric yet but must not be pathological) |
| **P4** | **Port GP3P (3-point) generalized absolute-pose minimal solver**, replacing the 6-point DLT for the new module's Path B (§3.3), and re-run the `metric_seed_candidates` threshold analysis with the new minimal-sample floor. | 1k, 2.5k, and the full 10k tier (COLMAP verified graph, matching M9's exact setup). | On the 10k tier: registered-frame-id set matches COLMAP's `4494`/`505` two-component split within ±1 frame each; Sim(3) scale within **10%** of `1.0`; ATE RMSE within **2x** of COLMAP's `0.305`–`0.384` m (M8/M9 control band). This is the first milestone directly comparable to the M9 result this plan starts from. | ~2-3 weeks (a real polynomial minimal solver + RANSAC integration + re-validation) |
| **P5** | **Full parity pass**: registration order LCS ≥0.9, points/reprojection within COLMAP's own run-to-run noise band, ATE RMSE ≤ COLMAP's own p95 band, on the full 10k tier. Only after P4 lands should any speed/memory deviation (matrix-free BA backends, sparsification, etc.) be reintroduced — and only with this harness gating that it does not regress the numbers below what P4 achieved. | 10k tier + at least one additional OpenLORIS sequence not used for tuning (held-out). | As stated above, on **two** sequences (one tuning, one held-out) to catch overfitting to corridor1-1's specific graph shape. | ~2-3 weeks + held-out validation |

**Start immediately with P1** — it is the smallest, most isolated, and most
directly evidenced fix: `frame.cc:85-94` is a ~10-line function, visloc
already has every other piece (P3P, `BaRigObservation`'s fixed-baseline
residual, `RigFrame`/`GeneralizedCameraRig`), and its absence is the single
cleanest "missing," not "deviates," row in §2's table.

---

## 6. Risks

- **Ceres vs. visloc LM numerical differences.** COLMAP's BA is Ceres
  Levenberg-Marquardt with Schur-complement direct/iterative solvers chosen
  by problem size (§1.6); visloc's `bundle.rs` LM implementation is a
  different, independently-written solver. Bit-exact parity is not a
  realistic goal (§4.2 already scopes tolerances, not exact equality, for
  everything past registered-frame-id sets); the risk is that a *correct*
  port could still show a several-percent scale/ATE gap purely from
  different convergence behavior (damping schedule, step acceptance,
  function/gradient tolerance defaults — COLMAP's are `function_tolerance=0`,
  `gradient_tolerance=1e-4`, `parameter_tolerance=0`,
  `bundle_adjustment_ceres.cc:103-105`) — this should be checked explicitly
  (run more LM iterations / tighter tolerances in visloc and confirm the
  parity gap shrinks) before concluding a milestone's remaining gap is a
  correctness bug rather than a convergence-depth difference.
- **RANSAC randomness.** Even with matching seeds, a different PRNG
  algorithm (COLMAP's vs. visloc's `SmallRng`/LCG-based samplers) will not
  produce the same hypothesis sequence — exact-inlier-set parity (§4.2 item
  1) should be treated as a nice-to-have diagnostic aid, not a hard gate;
  gate on outcomes (registered frame set, scale, ATE) instead.
- **License.** COLMAP core (`src/colmap/*`, including every file cited in
  §1) is BSD-3-Clause (ETH Zurich/UNC Chapel Hill) — a from-source Rust port
  of the algorithms in this document is legally clean, per
  `docs/colmap_port_plan.md`'s TL;DR (`docs/colmap_port_plan.md:24-33`).
  That note also flags two subdirectories to never port/depend on
  (`src/thirdparty/LSD`, AGPLv3; `src/thirdparty/SiftGPU`, non-commercial) —
  **neither is touched by this plan** (no feature extraction/matching code
  is in scope here, only the rig-aware mapper/BA).
- **Runtime.** Full-graph global BA (§3.3/P3) at 10k images / ~5000 frames
  is a materially larger dense/sparse Schur problem than the windowed
  backends `rig_sfm.rs` currently defaults to; COLMAP's own control run's
  `mapper` phase wall time and peak RSS are recorded in
  `tier-10000-rig-v3/result.json` (`phases.mapper`) and should be used as
  the *ceiling* to budget against, not a target to beat — this plan
  explicitly defers speed/memory optimization to after P5 (§5).
- **Regression risk to existing `rig_sfm.rs` callers.** Per §3.1, the new
  module is additive, not a rewrite of `rig_sfm.rs` — `RigTrackBuilder`
  variants like `MetricAnchoredCycle` may be intentionally conservative for
  reasons orthogonal to this benchmark (e.g. robustness on noisier
  real-world rigs than the OpenLORIS T265 pair). Any future
  decision to replace `rig_sfm.rs`'s default behavior with this port's
  should be a separate, explicitly-scoped follow-up, gated on the full
  existing `rig_sfm.rs` test suite (`rig_sfm.rs` has substantial inline
  `#[test]` coverage, e.g. the assertions at `rig_sfm.rs:7593-7594`) staying
  green.
