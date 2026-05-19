# VI Initialization → `OnlineSlamPipeline` Integration

Design note for wiring [`visloc_slam::VisualInertialInitializer`](../pipelines/slam/src/vi_initializer.rs) into [`OnlineSlamPipeline`](../pipelines/slam/src/lib.rs) as a pre-tracking bootstrap stage.

## Status

- Initialiser module: shipped and tested (20 unit tests, including time-weighted statistics, sliding-window detector, and the yaw-gauge-invariant `gravity_alignment_residual_deg` metric; empirical validation on 5 EuRoC sequences, see [progress.md](progress.md)).
- Pipeline integration: **shipped** in [`pipelines/slam/src/online_slam_vi_init.rs`](../pipelines/slam/src/online_slam_vi_init.rs) (+ wiring in [`pipelines/slam/src/lib.rs`](../pipelines/slam/src/lib.rs)). The design contract below is what the implementation matches; deviations would now appear as documentation drift and should be fixed in this file. Test coverage of the design points is in [`pipelines/slam/tests/online_slam.rs`](../pipelines/slam/tests/online_slam.rs) (`mod vi_init_integration`, 13 tests, 12 of the 14 design tests + 1 panic-on-construction variant). All 14 design tests are now shipped: test #12 standalone-validator constant-yaw invariance lives in [`pipelines/slam/src/vi_initializer.rs::tests::gravity_alignment_residual_is_invariant_under_world_yaw`](../pipelines/slam/src/vi_initializer.rs) — see [the test-strategy section](#test-strategy). Test #13 sliding-window detector is shipped on the standalone module — see [the detector-windowing section](#detector-windowing) below.
- Design review: this revision incorporates an external review (focus areas: pose-rotation convention, stale IMU factor handling, accel-bias observability, OSS-parity wording). The high-risk areas — `R_w←b → R_c←w` conversion with camera-center preservation, and stale factor discard — are pinned by explicit tests in the test-strategy section.

## Motivation

Today every IMU-aware path through `OnlineSlamPipeline` requires the caller to seed `OnlineSlamImuConfig.{bias_gyro, bias_acc, gravity_world}` and the first-keyframe pose **before** `new(map, tracker, mapper, config)` is called. The `examples/euroc_imu_dead_reckon_demo` example demonstrates this works: the same seed can be obtained from either Vicon ground truth (`--seed-from-gt`, the cheating path) or from `VisualInertialInitializer` (`--seed-from-vi-init`, the honest path). What the example does NOT demonstrate is the pipeline self-bootstrapping the seed from its own incoming IMU stream — every caller has to:

1. Buffer the leading IMU window out-of-band.
2. Construct a `VisualInertialInitializer`, call `push_sample` for each sample, call `try_initialize`.
3. Translate the returned `(bias_gyro, bias_acc, initial_rotation_body_to_world)` into `OnlineSlamImuConfig` and the first keyframe's pose.
4. Reset / reconstruct the pipeline with the new config.

This is the wrong shape for two reasons. First, the duplicated buffering means every downstream user re-implements the same boundary glue. Second, it forces the caller to know exactly when the visual frontend will hand the first keyframe — but the visual frontend lives behind the `Tracker` trait, which the caller does not own.

The integration pulls all of that into the pipeline: the caller pushes IMU samples through the existing `push_imu_measurement` entry point, the pipeline accumulates them in an internal initialiser buffer, and on the first frame where (a) the initialiser has accepted a stationary window AND (b) `process_frame` would have produced the first keyframe, the pipeline atomically promotes the recovered `(bias_gyro, bias_acc, R_w←b)` into `OnlineSlamImuState` and the first keyframe's pose.

## Non-Goals

The integration is deliberately **the stationary-window flavour only**. The following are explicit non-goals and should be addressed by separate work:

- **Motion-based VI initialisation.** ORB-SLAM3's "wait for translation, then run a motion-only optimisation" path recovers yaw and scale that the gravity-only stationary flavour cannot. That is a different algorithm, gated on a hot visual frontend, and worth its own design note.
- **Scale recovery for monocular-only pipelines.** The stereo / RGB-D / depth-prior path is already metric; the monocular path needs the motion-based flavour above.
- **Auto-updating `local_vi_ba_state.keyframe_state`.** Once the pipeline's linearisation point shifts, the sliding-window VI-BA already promotes per-keyframe `(velocity, bias_gyro, bias_acc)` slots from the running `OnlineSlamImuConfig` defaults on the next trigger; we do not need a separate hot-rewrite path.
- **Online re-initialisation after a long tracking loss.** The initialiser stays "first sequence only". Recovery from a tracking-lost segment is the relocalisation layer's responsibility.

## API surface

### New config

```rust
/// Optional auto-bootstrap stage that runs a `VisualInertialInitializer`
/// over the pipeline's incoming IMU stream and atomically promotes the
/// recovered `(R_w←b, b_g, b_a)` into the running pre-integrator on the
/// first keyframe. Requires `OnlineSlamConfig::imu = Some(_)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamViInitConfig {
    /// Inner stationary-window initialiser config (gravity, thresholds,
    /// minimum sample count, minimum window duration).
    pub initializer: VisualInertialInitializerConfig,
    /// Body-to-camera SE(3) extrinsic. Used to convert the initialiser's
    /// recovered `R_w←b` into the keyframe's `world-to-camera` Pose. When
    /// the IMU and camera share orientation, pass `SE3::identity()`. The
    /// promotion path is documented in `Coordinate convention checks`.
    pub body_to_camera: SE3,
    /// Whether to overwrite the first keyframe's `Pose` so its world→camera
    /// rotation reflects the recovered `R_w←b · R_b←c`. The camera center
    /// is preserved (the stationary-window flavour cannot observe absolute
    /// position); only the rotation is updated, with the translation
    /// recomputed from `t_c←w_new = -R_c←w_new · C_w_old`.
    pub seed_first_keyframe_rotation: bool,
    /// Behaviour when the initialiser fails to bootstrap before
    /// `max_wait_duration_seconds` of IMU has been buffered.
    pub on_persistent_rejection: ViInitFallback,
    /// Semantic cap: stop trying to initialise once this much wall-clock
    /// IMU duration has been buffered without success. Default `5.0 s`.
    /// `0.0` disables the duration cap.
    pub max_wait_duration_seconds: f64,
    /// Memory guard: refuse to buffer more than this many raw samples
    /// regardless of duration. Default `2000` (≈ 10 s @ 200 Hz). `0`
    /// disables the memory cap.
    pub max_buffered_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViInitFallback {
    /// Keep the pipeline's existing `OnlineSlamImuConfig` defaults; the
    /// caller has implicitly accepted that bias / rotation are not
    /// observable from this sequence's leading IMU window. The
    /// `OnlineSlamResult::vi_init` event carries the last rejection
    /// reason so downstream code can still log it.
    KeepExistingSeed,
    /// Disable IMU entirely for the rest of the sequence by setting
    /// `imu_state` and `local_vi_ba_state` to `None` after the cap is hit.
    /// Useful for "if VI init fails, fall back to visual-only" callers.
    DisableImuStage,
}
```

`OnlineSlamConfig` gains one field:

```rust
pub struct OnlineSlamConfig {
    pub apply_map_updates: bool,
    pub loop_closure: LoopClosureConfig,
    pub imu: Option<OnlineSlamImuConfig>,
    pub local_vi_ba: Option<OnlineSlamLocalBaConfig>,
    /// NEW. Requires `imu = Some(_)`; ignored when `imu` is `None`.
    pub vi_init: Option<OnlineSlamViInitConfig>,
}
```

### New running state

`OnlineSlamPipeline` gains one mirror field, following the existing `imu_state` / `local_vi_ba_state` pattern. Unlike the existing `pub imu_state` / `pub local_vi_ba_state` fields, the new `vi_init_state` is **private** because writes to it (especially `completed`) cross-cut with `imu_state` and with `map.keyframes` invariants — exposing it as `pub` would let callers leave the pipeline in a half-initialised state. Read-only inspection goes through the `vi_initialization_status()` accessor:

```rust
pub struct OnlineSlamPipeline<T, M> {
    pub map: VisualMap,
    pub tracker: T,
    pub mapper: M,
    pub config: OnlineSlamConfig,
    pub imu_state: Option<OnlineSlamImuState>,
    pub local_vi_ba_state: Option<OnlineSlamLocalBaState>,
    // NEW. Private. `Some` exactly when
    // `config.vi_init.is_some() && config.imu.is_some()`. Inspected via
    // `vi_initialization_status()`; written exclusively by `process_frame`.
    vi_init_state: Option<OnlineSlamViInitState>,
}

/// Private. Owned only by `OnlineSlamPipeline` so the `completed` /
/// `imu_state` / `map.keyframes` invariants cannot drift apart.
struct OnlineSlamViInitState {
    config: OnlineSlamViInitConfig,
    initializer: VisualInertialInitializer,
    /// Set once the initialiser has succeeded; subsequent IMU samples
    /// stop being forwarded into the buffer AND `take_pending_imu_factor`
    /// starts returning the regular pre-integration factors.
    completed: Option<VisualInertialInitializationResult>,
    /// Set when `on_persistent_rejection` has fired. Captures the last
    /// rejection so downstream observers can see it.
    gave_up: Option<StationaryRejectionReason>,
    /// Running counter — bumped on every `push_imu_measurement` call until
    /// `completed` or `gave_up` is `Some`.
    samples_buffered: usize,
    /// Running buffered duration — used to enforce
    /// `max_wait_duration_seconds`. Updated alongside `samples_buffered`.
    buffered_duration_seconds: f64,
}

/// Read-only snapshot exposed to callers.
#[derive(Debug, Clone, PartialEq)]
pub enum ViInitializationStatus {
    /// IMU + VI init both disabled, or only IMU is enabled (no auto-bootstrap).
    Disabled,
    /// Buffering samples; `try_initialize` has not yet succeeded.
    Buffering {
        samples_buffered: usize,
        buffered_duration_seconds: f64,
        last_rejection: Option<StationaryRejectionReason>,
    },
    /// `try_initialize` has succeeded; `result` reflects what was promoted
    /// into `imu_state` (and into the first keyframe's pose if
    /// `seed_first_keyframe_rotation` was set).
    Initialised { result: VisualInertialInitializationResult },
    /// Cap exceeded; the configured fallback has already been applied.
    GaveUp { last_reason: StationaryRejectionReason, fallback: ViInitFallback },
}

impl<T, M> OnlineSlamPipeline<T, M> {
    pub fn vi_initialization_status(&self) -> ViInitializationStatus { /* … */ }
}
```

### Result surface

```rust
pub struct OnlineSlamResult {
    pub tracking: TrackingResult,
    pub mapping: Option<MappingResult>,
    pub applied_update: Option<AppliedMapUpdate>,
    pub loop_closure_candidates: Vec<LoopClosureCandidate>,
    pub imu_factor: Option<ImuPreintegrationFactor>,
    pub local_vi_ba: Option<OnlineSlamLocalBaStats>,
    pub map_keyframe_count: usize,
    pub map_landmark_count: usize,
    /// NEW. State-transition event for VI init. `Some` ONLY on the frame
    /// where the initialiser actually transitioned state — `Succeeded`,
    /// `StillBuffering` (a non-fatal `try_initialize` rejection this
    /// frame), or `GaveUp`. `None` once `Succeeded` or `GaveUp` has been
    /// emitted (those are terminal); also `None` on frames where the
    /// initialiser was not run (e.g. no new keyframe). Callers that need
    /// the durable state should use `vi_initialization_status()`.
    pub vi_init: Option<ViInitializationEvent>,
}

pub enum ViInitializationEvent {
    /// `try_initialize` succeeded this frame. The pipeline's running
    /// `OnlineSlamImuState.preintegrator` has been reset with the new
    /// bias linearisation; the first keyframe's pose has been updated
    /// if `seed_first_keyframe_rotation` is true; stale IMU factors
    /// staged before the seed promotion have been discarded
    /// (see `Behavioural contract`). Emitted at most once per sequence.
    Succeeded {
        result: VisualInertialInitializationResult,
        first_keyframe_id: Option<u64>,
        discarded_stale_factor_count: usize,
    },
    /// `try_initialize` was attempted this frame and was rejected.
    /// The buffer is preserved; the initialiser will try again on the
    /// next attempt. Carries the failing predicate's measured value
    /// so downstream code can decide whether to widen the thresholds.
    StillBuffering { reason: StationaryRejectionReason },
    /// `max_wait_duration_seconds` (or `max_buffered_samples`) was reached
    /// without success. The pipeline applied `on_persistent_rejection`.
    /// Emitted at most once per sequence; subsequent frames see
    /// `vi_init: None`.
    GaveUp { last_reason: StationaryRejectionReason, fallback: ViInitFallback },
}
```

### Behavioural contract

- `push_imu_measurement(gyro, accel, dt)`: when `vi_init_state` is `Some(s)` and `s.completed.is_none() && s.gave_up.is_none()`, the sample is also pushed into `s.initializer` via the existing `VisualInertialInitializer::push_sample` (non-positive `dt` is silently dropped at both layers — same convention as `ImuPreintegrator::integrate_sample`). `s.samples_buffered` and `s.buffered_duration_seconds` are updated. The existing IMU pre-integrator also ingests the same sample so the running pre-integration window is not stalled — but see the **stale factor gate** below for what happens to those factors before init succeeds.
- **Stale factor gate.** Until `s.completed.is_some()`, `take_pending_imu_factor()` returns `None` and the inline `OnlineSlamResult.imu_factor` is suppressed even when a keyframe transition would normally stage one. Rationale: factors staged before the seed promotion were built with the *placeholder* bias linearisation from `OnlineSlamImuConfig` defaults and would feed an inconsistent bias point into the VI-BA stage if released. Once init succeeds, those buffered raw IMU samples are dropped (NOT re-integrated — the MVP design picks "discard old, start fresh" over "re-integrate with new bias linearisation" because the dropped window is at most `max_wait_duration_seconds` of pre-keyframe motion that the visual frontend has not yet attached anything to). `discarded_stale_factor_count` on the `Succeeded` event reports how many factors were dropped so callers can audit.
- `process_frame(frame, candidates)`: after the existing tracking / mapping / loop-detection / IMU-factor staging, the pipeline runs a single try-initialise step IFF `vi_init_state.is_some()` AND `s.completed.is_none() && s.gave_up.is_none()` AND a NEW keyframe was just registered (the same condition that already gates `stage_imu_factor_on_new_keyframe`). Three cases:
  - **Success.** Atomic promotion in this order:
    1. `OnlineSlamImuState.preintegrator` is reset via `ImuPreintegrator::new_with_bias(result.bias_gyro, result.bias_acc)`. Any stale pending factor is dropped (counted into `discarded_stale_factor_count`).
    2. `OnlineSlamImuConfig.{bias_gyro, bias_acc}` are mirrored to the new values so the next sliding-window VI-BA promotion uses them as its `bias_*_init`.
    3. If `seed_first_keyframe_rotation` is true, the just-registered keyframe's `Pose` is rewritten so its rotation reflects the recovered orientation while the camera center is preserved (see `Coordinate convention checks` for the exact formula).
    4. `OnlineSlamLocalBaState.keyframe_state` is seeded for the first keyframe with `(velocity_world: 0, bias_gyro, bias_acc)`.
    5. `vi_init_state.completed = Some(result)`. `OnlineSlamResult.vi_init = Some(Succeeded { ... })`.
  - **Rejection but still trying.** Two caps are checked in this order: `s.buffered_duration_seconds >= max_wait_duration_seconds` (semantic) and `s.samples_buffered >= max_buffered_samples` (memory). Either triggers the `GaveUp` fallback. Otherwise nothing else changes; `OnlineSlamResult.vi_init = Some(StillBuffering { ... })`.
  - **GaveUp / fallback.** `vi_init_state.gave_up = Some(reason)`. `KeepExistingSeed` leaves `imu_state` and `local_vi_ba_state` untouched and lifts the **stale factor gate** (`OnlineSlamImuConfig` defaults were the caller's chosen seed all along; pre-buffered factors stay discarded but new factors flow). `DisableImuStage` sets both `imu_state` and `local_vi_ba_state` to `None` and clears `config.imu` / `config.local_vi_ba`. `OnlineSlamResult.vi_init = Some(GaveUp { ... })`.
- `reset_sequence_state()`: also calls `VisualInertialInitializer::reset` on the inner initialiser and clears `completed` / `gave_up` / `samples_buffered` / `buffered_duration_seconds`. The next sequence gets a fresh bootstrap and the stale factor gate re-arms.

### Construction

```rust
let mut pipeline = OnlineSlamPipeline::new(
    map,
    tracker,
    mapper,
    OnlineSlamConfig {
        imu: Some(OnlineSlamImuConfig {
            gravity_world: Vector3::new(0.0, 0.0, -9.81),  // EuRoC z-up
            // bias seeds are "good enough" defaults; vi_init will overwrite them
            ..Default::default()
        }),
        vi_init: Some(OnlineSlamViInitConfig {
            initializer: VisualInertialInitializerConfig {
                gravity_world: Vector3::new(0.0, 0.0, -9.81),
                min_stationary_window_seconds: 0.5,
                max_gyro_std: 0.05,
                max_accel_std: 0.5,
                max_accel_magnitude_error: 0.5,
                min_samples: 50,
            },
            body_to_camera: SE3::identity(),  // IMU and camera share orientation
            seed_first_keyframe_rotation: true,
            on_persistent_rejection: ViInitFallback::KeepExistingSeed,
            max_wait_duration_seconds: 5.0,
            max_buffered_samples: 2000,  // memory guard
        }),
        ..Default::default()
    },
);
```

Construction-time check: `OnlineSlamPipeline::new` panics with a descriptive message when `config.vi_init.is_some() && config.imu.is_none()`. This is a developer error, not a runtime condition — and "panic on construction" is consistent with how the existing `local_vi_ba` requires `imu = Some(_)` (which it currently encodes as a doc comment; we should tighten both at the same time during the integration).

The two `gravity_world` values in the example above are deliberately the same on both `imu` and `vi_init.initializer`. The integration's first PR should add a `OnlineSlamConfig::validate(&self) -> Result<(), ConfigError>` debug helper that asserts they match, plus the gyro/accel std limits are positive and the min duration is reasonable.

## Coordinate convention checks

This section is load-bearing. The single highest-risk part of the integration is the rotation-direction promotion: getting it slightly wrong silently produces world-frame inversion and tens-to-hundreds of metres of ATE — and the bug looks identical to "your VI init is bad", so an integration with a wrong sign cannot be told apart from an algorithmic problem until the convention is pinned by tests.

- **Gravity.** `OnlineSlamImuConfig.gravity_world` and `VisualInertialInitializerConfig.gravity_world` MUST agree. Both are world-frame, both default to KITTI y-down on the `OnlineSlamImuConfig` side and EuRoC z-up on the initialiser side. The integration should NOT silently coerce one into the other — assertion on `OnlineSlamPipeline::new` is the right behaviour. (This is why `gravity_world` is duplicated rather than read once: the initialiser is also used standalone by callers that never construct an `OnlineSlamConfig`.)
- **Body-frame samples.** Both `ImuPreintegrator::integrate_sample` and `VisualInertialInitializer::push_sample` consume body-frame angular velocity / linear acceleration with the *same* convention (no gravity pre-subtraction). The pipeline forwards the same sample to both layers untouched.
- **Rotation direction (READ CAREFULLY).** The initialiser returns `R_w←b ≡ initial_rotation_body_to_world` — the rotation that maps a body-frame vector into the world frame (also written `R_wb`). The keyframe `Pose` in `visloc-core` is `T_c←w` (`world-to-camera`), so its `rotation` field is `R_c←w = R_w←c^T`. The promotion path therefore does NOT write `R_w←b` directly into `Pose.rotation`. The correct chain is:
  ```text
  R_w←c = R_w←b · R_b←c          // R_b←c from OnlineSlamViInitConfig.body_to_camera
  R_c←w = R_w←c^T                 // this is what goes into Pose.rotation
  ```
  When the IMU and camera share orientation (`R_b←c = I`), this collapses to `R_c←w = R_w←b^T` — i.e. **the transpose of the initialiser's output**, NOT the output itself. Writing `R_w←b` directly into a world-to-camera `Pose.rotation` would put the world upside down relative to the camera, and the resulting dead-reckon + visual-aided ATE would inflate by hundreds of metres before tracking failed and masked the bug.
- **Camera center preservation.** `Pose.translation` is `t_c←w`, NOT the camera position in world. Overwriting `rotation` while leaving `translation` untouched changes the camera center, because `C_w = -R_c←w^T · t_c←w` depends on the rotation. The promotion path preserves `C_w` by also recomputing `t_c←w`:
  ```text
  C_w_old      = -R_c←w_old^T · t_c←w_old
  R_c←w_new    = (R_w←b · R_b←c)^T
  t_c←w_new    = -R_c←w_new · C_w_old
  Pose_new     = Pose::from_world_to_camera(R_c←w_new, t_c←w_new)
  ```
  The stationary-window flavour cannot observe absolute position, so we keep whatever camera center the tracker / mapper already produced for the first keyframe.
- **Velocity.** The initialiser sets `initial_velocity_world = 0` by construction (stationary window). When the pipeline promotes the seed, `OnlineSlamLocalBaState.keyframe_state[first_keyframe_id]` is seeded with `velocity_world = 0` so the first VI-BA trigger does not bootstrap from a non-zero velocity. The current `online_slam_vi_ba.rs` promotion path seeds new keyframes from `bias_gyro_init` / `bias_acc_init` on `OnlineSlamLocalBaConfig`; we add a parallel "first-keyframe velocity seed" hook, default zero.

### Accel-bias caveat

`b_a` returned by the initialiser is *mostly* the **magnitude residual** between the mean specific force and `‖g_w‖`, not a full 3-axis accel bias. Because `R_w←b` is constructed from `ā_b` itself, any lateral component of the true accel bias is absorbed into the recovered rotation (it tilts the seed to align `ā_b` with world-up); only the component parallel to `ā_b` survives into `b_a`. In practice this means:

- If your IMU has a strong gravity-axis accel bias, the initialiser captures it well.
- If your IMU has a strong cross-axis accel bias (`b_a_x`, `b_a_y` in body frame, with gravity along body-z), the initialiser will *not* recover it; it will appear as roll/pitch error in `R_w←b`.

The pipeline integration documents this so callers don't expect more bias observability from the static bootstrap than the algorithm can give. The sliding-window VI-BA stage downstream is the right place to recover the full 3-axis accel bias because translation in the visual frontend gives the missing observability axes.

## Failure-mode reasoning

The stationary-window initialiser has five failure modes (one for each `StationaryRejectionReason` variant). The pipeline integration handles them uniformly: the buffer is preserved and the next sample push triggers another attempt. The caller controls when to give up via `max_wait_duration_seconds` / `max_buffered_samples` and what to do via `on_persistent_rejection`. Empirically (see [progress.md](progress.md)):

- `InsufficientSamples` / `InsufficientDuration`: transient. Always self-resolves with more samples. Default `max_wait_duration_seconds = 5.0` is well above the EuRoC stationary holding period of ~ 1 s, while `max_buffered_samples = 2000` (≈ 10 s @ 200 Hz) gives a generous memory ceiling.
- `GyroNoiseTooHigh` / `AccelNoiseTooHigh`: persistent on sequences where the leading IMU window contains real motion (`MH_01_easy`, `V1_01_easy`). The fallback is the right escape hatch — neither widening the threshold nor waiting longer would help.
- `AccelMagnitudeMismatch`: indicates the gravity vector is wrong (sign / axis) or the IMU calibration is off. Should be treated as a hard configuration error in production; the integration logs it via `GaveUp` and the caller should surface it to the user.

The integration does NOT auto-widen thresholds when buffering. That decision belongs to the caller — they have more context about the dataset.

## Test strategy

All tests use synthetic IMU streams and stub / no-op visual frontends — synthetic inputs are the right tool here because the tests target pipeline-state-machine correctness and pose-convention correctness, not benchmark numbers. The EuRoC validation already done on the standalone initialiser covers the "real dataset" side; re-running it through the pipeline only changes the call path, not the recovered values.

The list below is **the minimum coverage required to land the integration**. Tests marked `[pose-conv]` exercise the load-bearing rotation/translation conversion and MUST be present before merge — the convention bug they catch is otherwise invisible until a real dataset run.

1. **Config validation.** `vi_init: Some(_), imu: None` → `OnlineSlamPipeline::new` returns `ConfigError::ViInitRequiresImu`.
2. **Gravity mismatch.** `vi_init.initializer.gravity_world != imu.gravity_world` → `ConfigError::GravityMismatch`.
3. **Stationary-stream success.** Drive `push_imu_measurement` + `process_frame` with a known-good stationary stream + a single fake keyframe; assert `OnlineSlamResult.vi_init = Some(Succeeded { .. })`, assert `imu_state.preintegrator`'s linearisation point swapped, assert the recovered `(bias_gyro, bias_acc, rotation)` matches the synthesised ground truth within `1e-9`.
4. **Succeeded event fires at most once.** After the success frame, subsequent `process_frame` calls return `vi_init: None`; `vi_initialization_status()` returns `Initialised { .. }`.
5. **Stale factor discard.** Seed the pipeline so 3 IMU factors stage *before* the success frame; assert `take_pending_imu_factor()` returns `None` while `vi_init.completed.is_none()`; assert the `Succeeded` event reports `discarded_stale_factor_count == 3`; assert subsequent factors flow normally.
6. **[pose-conv] Rotation direction.** Synthesise a stationary IMU window for a body tilted 30° about world-x. With `body_to_camera = SE3::identity()`, after success the keyframe `Pose.rotation` is `R_c←w = (rotation_about_x(30°))^T`, NOT `rotation_about_x(30°)`. Apply the resulting Pose to the world-up vector and assert it lands on `[0, 0, 1]` in camera frame within `1e-9`.
7. **[pose-conv] Camera center preservation.** Set the first keyframe's pre-success Pose to a known `(R_c←w_old, t_c←w_old)`. After success, assert `-R_c←w_new^T · t_c←w_new == -R_c←w_old^T · t_c←w_old` within `1e-9` (the camera center is unchanged).
8. **[pose-conv] Body-to-camera extrinsic.** Synthesise a stationary stream with body tilted 30° about world-x AND a non-identity `body_to_camera = SE3::rotation_about_y(90°)`. Assert the final `Pose.rotation` matches `(R_w←b · R_b←c)^T` within `1e-9`.
9. **Failure-mode StillBuffering.** Drive noisy gyro (alternating ±1 rad/s); assert frames carry `StillBuffering { reason: GyroNoiseTooHigh { .. } }` until the cap.
10. **`KeepExistingSeed` fallback.** Exceed `max_wait_duration_seconds`; assert `GaveUp { fallback: KeepExistingSeed }` is emitted, `imu_state` is still `Some`, and from this frame on the stale-factor gate is lifted (new factors flow with the original `OnlineSlamImuConfig` defaults).
11. **`DisableImuStage` fallback.** Same trigger as above; assert `imu_state` and `local_vi_ba_state` are both set to `None`, and `config.imu` / `config.local_vi_ba` are cleared.
12. **Constant-yaw invariance.** ✅ Shipped on the standalone module — see [`pipelines/slam/src/vi_initializer.rs::tests::gravity_alignment_residual_is_invariant_under_world_yaw`](../pipelines/slam/src/vi_initializer.rs). The metric is exposed as the public method `VisualInertialInitializationResult::gravity_alignment_residual_deg(&UnitQuaternion<f64>) -> f64` so callers (the [`examples/euroc_imu_dead_reckon_demo`](../examples/euroc_imu_dead_reckon_demo.rs) ablation harness, the standalone validator, future EuRoC ATE harnesses) all read the same yaw-gauge-aware residual. The test builds two stationary streams whose ground-truth body→world rotations differ only by a 30° rotation about the world "up" axis and asserts `gravity_alignment_residual_deg ≈ 0°` for both, while the unfiltered quaternion residual against the yaw-rotated stream is ~30° — proving the metric correctly identifies yaw as gauge instead of charging it against the initialiser.
13. **Sliding-window non-stationary→stationary.** ✅ Shipped on the standalone module — see [`pipelines/slam/src/vi_initializer.rs::tests::sliding_window_non_stationary_then_stationary_succeeds`](../pipelines/slam/src/vi_initializer.rs). With `detector_window_seconds = 1.0`, 0.5 s of noisy gyro followed by 1.5 s of stationary samples succeeds and `samples_consumed` reflects only the trailing 200-sample window. See [the detector-windowing section](#detector-windowing) for the full design.
14. **`reset_sequence_state` re-arms.** After a successful initialisation, call `reset_sequence_state`; assert `vi_initialization_status()` returns to `Buffering { samples_buffered: 0, .. }` and the next stationary stream can re-bootstrap.

EuRoC validation of the pipeline-integrated path is shipped as two companion demos:

* [`examples/euroc_online_slam_vi_demo`](../examples/euroc_online_slam_vi_demo.rs) — drives `OnlineSlamPipeline` with the real ~200 Hz EuRoC IMU stream and cam0 frame cadence, with `vi_init: Some(_)` enabled; cam0 image pixels are intentionally **not** decoded — instead the demo seeds a deterministic 5×5 landmark grid in front of the first GT camera pose and projects it into each cam0 frame under the GT-derived `R_c←w = (R_w←b · R_b←c)^T` (with cam0's `T_BS` honoured), so the visual side is anchored and the integration question being answered is "does the pipeline + auto-bootstrap promotion work on a real EuRoC IMU + cam0 timestamp stream".
* [`examples/euroc_online_slam_vi_image_demo`](../examples/euroc_online_slam_vi_image_demo.rs) — the natural next step: decodes every cam0 PNG with `visloc-io`'s `read_common_image` (gated on the `image-io` feature) and runs `CornerFeatureExtractor` on it, feeding real corner keypoints + patch descriptors into `process_frame`. The visual map is bootstrapped by extracting corners on both cam0 *and* cam1 for the seed frame, matching descriptors across the stereo pair, and triangulating each surviving match via [`visloc_vision::stereo_bootstrap::bootstrap_stereo_landmarks`](../crates/vision/src/stereo_bootstrap.rs) using the published `T_BS` extrinsics — each triangulated landmark is metric-scale, so the seed map no longer collapses every corner onto a single `--bootstrap-depth` shell. Corners that fail to match a cam1 descriptor still fall back to the `--bootstrap-depth` back-projection so the tracker has consistent keypoint coverage from frame zero; pass `--no-stereo-bootstrap` to disable the stereo pass entirely (depth-only seeding, for A/B comparison). Cam0 and cam1 radial-tangential distortion are both applied by default via [`visloc_vision::distortion::RadialTangential`](../crates/vision/src/distortion.rs) (parsed from each camera's published `distortion_coefficients`); pass `--no-undistort` to reproduce the pre-correction behaviour. The tracker's default `BruteForceMatcher` + `PnPRansac` match real descriptors from frame zero. This demo answers a harder integration question: "does the pipeline + auto-bootstrap stage survive a real pixel-derived feature stream (variable corner counts, real descriptor signal-to-noise, tracking dropouts) on a real EuRoC sequence". Two scope cuts have now been closed (per-keypoint distortion correction, stereo-triangulation seed map); the remaining "planned" line in the OSS comparison table below is motion-based / dynamic VI alignment — see [`docs/motion_based_vi_alignment.md`](./motion_based_vi_alignment.md) for the design note.

Both demos' outputs (`slam_trajectory.csv`, `slam_errors.csv`, `vi_init_log.txt`, `summary.txt`) mirror the shape of `examples/euroc_imu_dead_reckon_demo` so they can be diffed apples-to-apples on the same EuRoC sequence.

## Detector windowing

**Status: shipped.** `VisualInertialInitializerConfig.detector_window_seconds: f64` (default `f64::INFINITY`) now controls a trailing sliding window for the stationary-window predicates and statistics. When the buffer exceeds `detector_window_seconds`, only the trailing slice — walked from the END of the buffer backwards until cumulative `dt` first reaches the configured width — is evaluated. The default `INFINITY` preserves the historical "evaluate on the whole buffer" behaviour exactly, so existing callers see zero numerical change.

Why the change matters: the original "all-buffer" evaluation was right for the standalone validation harness (EuRoC's leading hold is at the very start of the sequence), but the pipeline-integrated path widens the use case — a caller might start the pipeline during motion and only become stationary later. With "all-buffer" statistics, early motion permanently spoils the std/magnitude tests; the buffer can never re-enter the stationary regime even after the body actually settles. The sliding window restores the trailing-clean-window detection that OpenVINS's `StaticInitializer` uses.

The mean / variance is also now **time-weighted**: `μ = Σ x_i · dt_i / Σ dt_i`, `σ² = Σ (x_i - μ)² · dt_i / Σ dt_i`. For uniform `dt` this collapses to the historical sample-count weighted formula exactly (every existing test still passes with zero modification); for irregular `dt` from a real IMU stream that drops or duplicates samples, the time-weighted form removes a small but real bias in the statistics.

Test coverage in [`pipelines/slam/src/vi_initializer.rs`](../pipelines/slam/src/vi_initializer.rs) (`mod tests`):

* `default_detector_window_seconds_is_infinity` — backward-compat anchor.
* `time_weighted_statistics_match_unweighted_for_uniform_dt` — pins the property that under uniform `dt` the weighted formula matches the historical output exactly.
* `time_weighted_statistics_shift_mean_under_nonuniform_dt` — under non-uniform `dt`, the weighted mean tracks the time-weighted form (much closer to the high-weight sample, not the sample-count midpoint).
* `sliding_window_non_stationary_then_stationary_succeeds` — **test #13 from the test-strategy section, now unblocked.** 0.5 s of noisy gyro followed by 1.5 s of stationary samples with `detector_window_seconds = 1.0` succeeds, and `samples_consumed` reflects only the trailing 200 samples (entirely from the stationary phase).
* `sliding_window_rejects_when_recent_samples_are_noisy` — the reverse path: 1.5 s stationary then 0.5 s noisy gyro with `detector_window_seconds = 0.5` rejects with `GyroNoiseTooHigh`, confirming the slice is taken from the END of the buffer.
* `sliding_window_insufficient_when_trailing_slice_too_short` — predicates apply to the SLICE, not the total buffer: a 400-sample buffer with a 0.1 s window and `min_stationary_window_seconds = 0.5` is rejected as `InsufficientDuration`.

## Design alternatives considered

### Nested `ImuBootstrapConfig` enum

The reviewer suggested folding `vi_init` into `OnlineSlamImuConfig` as a nested enum:

```rust
pub struct OnlineSlamImuConfig {
    pub gravity_world: Vector3<f64>,
    pub bias_gyro: Vector3<f64>,
    pub bias_acc: Vector3<f64>,
    pub bootstrap: ImuBootstrapConfig,
    // ...
}

pub enum ImuBootstrapConfig {
    Disabled,
    StaticWindow(OnlineSlamStaticViInitConfig),
    // future:
    // DynamicAlignment(OnlineSlamDynamicViInitConfig),
    // StaticThenDynamic { static_init: ..., dynamic_init: ... },
}
```

This *is* tighter from a type-state perspective — invalid combinations like `vi_init: Some(_), imu: None` or `gravity_world` mismatch become impossible to express. The trade-off is API breaking: every existing caller of `OnlineSlamImuConfig` needs to add a `bootstrap: ImuBootstrapConfig::Disabled` field.

**Decision: keep the `Option<OnlineSlamViInitConfig>` shape for now** because (a) it preserves backward compatibility on `OnlineSlamConfig` (the typical entry point), (b) `vi_init` is naturally a sibling of `local_vi_ba` (both are opt-in IMU-dependent stages), and (c) the validation gap can be closed with a `OnlineSlamConfig::validate(&self) -> Result<(), ConfigError>` helper that runs on `new`. If a `DynamicAlignment` flavour later turns out to be more naturally expressed as part of the same bootstrap state machine, we revisit and migrate to the nested enum at that point.

### Re-integrate stale samples instead of discarding

The behavioural contract drops pre-success IMU factors rather than re-integrating the raw buffered samples with the new bias linearisation. The alternative — buffer raw samples in a side channel, then on success, re-run the pre-integrator with the new bias to rebuild factors retroactively — would preserve more pose-graph constraints across the bootstrap boundary. We reject this for the MVP because (a) the pre-keyframe window has nothing visual attached to it yet, so the dropped factors connect *nothing* on the BA side, (b) the buffered samples are at most `max_wait_duration_seconds = 5.0 s` of motion that the visual frontend will revisit through normal tracking once init is up, and (c) the implementation complexity of "re-integration with new linearisation" is a maintenance hazard for a corner that touches no real signal. The `discarded_stale_factor_count` audit field on the `Succeeded` event is the explicit knob for a caller that wants to detect the drop and run a separate recovery.

## OSS comparison and feature-matrix wording

The feature matrix in the standalone validator currently reads "VI initialization: ○" for `visloc-rs` after this work lands. The reviewer correctly flags that this is comparable to **OpenVINS's `StaticInitializer` / OKVIS's "boot-strapping at rest"**, not to **ORB-SLAM3's full multi-stage VI initialisation** which also runs an inertial-only MAP optimisation (`VIBA1` / `VIBA2`) over the first 5–15 s of trajectory and continues to refine scale on monocular sequences.

After this integration ships, the honest matrix entry is:

| System | Static VI bootstrap | Dynamic VI alignment | Periodic scale refinement |
|--|--|--|--|
| visloc-rs | ○ (this work) | ○ shipped — VIBA1 (inertial-only) + VIBA2 (inertial-with-scale outer loop, monocular scale recovery via closed-form 1-D LS) shipped via [`MotionBasedViInitializer`](../pipelines/slam/src/vi_motion_initializer.rs) + pipeline glue [`OnlineSlamMotionViInitConfig`](../pipelines/slam/src/online_slam_motion_vi_init.rs); see [`motion_based_vi_alignment.md`](./motion_based_vi_alignment.md) | partial (existing `local_vi_ba` already refines scale-equivalent on stereo) |
| ORB-SLAM3 | ○ | ○ (`VIBA1`/`VIBA2`) | ○ |
| VINS-Mono | ○ | ○ (SfM + linear alignment) | ○ |
| OKVIS | ○ | n/a (stereo, scale is metric) | n/a |
| Kimera-VIO | factor-graph bootstrap | ○ | n/a |

This framing keeps the parity claim honest. With VIBA1 + VIBA2 + pipeline glue now landed, the remaining work in this column is empirical tuning on EuRoC and TUM-VI (rather than fresh design or solver work) and an eventual upgrade of the VIBA2 inner solve from the current "inertial-only + outer scale loop" to a fully joint visual-inertial-scale BA (where the visual landmarks + scale are co-optimised inside one normal-equations system rather than alternating). The alternating formulation is correct for synthetic constant-velocity scenes; the joint formulation closes a more general convergence gap on monocular sequences with weakly-informative kinematics.

## Backward compatibility

`OnlineSlamConfig::vi_init` defaults to `None`. Every existing caller — including the IMU-aware `examples/euroc_imu_dead_reckon_demo`, the synthetic VI-BA integration tests, and the appearance-only `examples/online_slam_loop_candidate_dummy` — keeps the same `OnlineSlamResult` shape (with `vi_init: None` always). The new `vi_init_state: Option<…>` field on `OnlineSlamPipeline` is `None` whenever `config.vi_init.is_none()`, mirroring the `imu_state` and `local_vi_ba_state` pattern.

No public API on `VisualInertialInitializer` itself changes. The integration is purely additive on `visloc_slam` and the `OnlineSlam*` types; the `prelude` re-exports already cover the new types after the integration lands (see [interfaces.md](interfaces.md)).
