# Monocular-inertial VIO on the Basalt port (design / status)

**Status: experimental, opt-in.** It runs end to end and is verified on
synthetic data, including rendered images through the real KLT frontend.
**No EuRoC ATE has been measured yet** because the machine that did this work
cannot reach the dataset. The stereo-inertial path is unchanged.

## Summary

Upstream Basalt's `SqrtKeypointVioEstimator` (pinned commit `0f3b2b52`) works
with any number of cameras. Single-camera support therefore needed **no
estimator change**. The investigation below shows that the port already
follows upstream in every place where the camera count matters. The gap was
plumbing: nothing could open a EuRoC recording, or build the adapter, as
cam0 + IMU. This change adds that plumbing behind an explicit opt-in. It also
adds synthetic tests showing that the monocular estimator tracks and recovers
metric scale from the IMU.

## Investigation: where the camera count matters

| Concern | What the port does | Mono result |
| --- | --- | --- |
| Calibration (`calibration.rs`) | Accepts 1..N cameras. Intrinsics, extrinsics and resolutions are validated to equal length. | Already worked. Added `BasaltCalibration::retain_cameras(n)` to turn a stereo file into a cam0-only rig. |
| Frontend (`stream.rs`, `DirectKltStream`) | `StereoFrame::cam1` is `Option`. The stereo KLT, stereo FB² and essential-matrix stages run only if a cam1 pyramid *and* a camera-1 calibration both exist. A cam1 image without a camera-1 calibration is an error. | Already worked: cam0 temporal KLT and FAST replenishment, with no stereo stage. |
| EuRoC reader (`euroc.rs`) | Reads the `cam1` manifest and keeps only cam0 frames whose timestamp also appears in cam1, as upstream's stereo optical flow does. | **Gap**: `cam1` was required. Added `open_monocular[_with_timing]`. |
| Estimator construction (`adapter.rs`) | `with_camera_rig(calibration.cameras, calibration.t_imu_cam)`. | Already generic. The construction was extracted into `vio_estimator_from_calibration` (same code) so tests can use it. |
| Landmark creation (`estimator.rs`, keyframe branch) | Same as upstream: the host is `TimeCamId(current, 0)`, and candidates are **every retained observation of the track**, ordered as upstream's `std::map<TimeCamId,…>` (another camera at the same time, or any camera at an earlier frame). The first candidate whose IMU-propagated baseline passes the 5 cm gate is triangulated. | Already worked: with one camera every candidate is an earlier cam0 frame, which is temporal (multi-frame) triangulation. |
| Initialization (`initial_nav_from_imu`) | Same as upstream `initialize()`: roll and pitch come from the first accelerometer sample at or after the first frame. Yaw, position, velocity and biases start at zero. The prior covers position, yaw and biases; velocity is left free. | Already worked. It needs no stereo baseline: the first keyframe hosts no landmarks, and later keyframes triangulate against the IMU-propagated poses, so landmark depth (and therefore scale) is metric. |
| Keyframe policy (`decide_keyframe`) | Counts cam0 tracks that are connected to a landmark versus all cam0 tracks. | Unchanged. While no landmark exists the ratio is 0, so a keyframe is taken as soon as `frames_after_kf > vio_min_frames_after_kf`. |
| Marginalization / MargData | Built from the active window's pose/state blocks and per-camera `OfImageData`. | Unchanged. Mono MargData passes `validate_contract`, and mapper packets are emitted on keyframe marginalization (tested). |
| Mapper / online SLAM (`mapper/`, `basalt_euroc_online_slam_demo`) | The NFR mapper has a stereo matching stage. | **Not wired for mono.** `--mono` exists only on the VIO demo. |

Related work elsewhere in the repo, not duplicated here:

- `docs/dpvo_droid_port_plan.md` M5 adds IMU coupling to the DPVO window
  (`pipelines/slam/src/dpvo_vi_ba.rs`). That is a separate learned-frontend
  system, and its bootstrap is documented there as not yet reliable.
- `pipelines/slam` has a motion-based VI initializer (VIBA1/VIBA2, see
  `docs/motion_based_vi_alignment.md`) for the generic online SLAM pipeline.

The Basalt path does not use either of them.

## What changed

All changes are additive. The default and stereo code paths are unchanged
(see the next section).

- `BasaltCalibration::retain_cameras(count)`: returns a copy that keeps the
  first `count` cameras.
- `EurocSensorDataset::open_monocular` and `open_monocular_with_timing`:
  - never read `mav0/cam1` (the directory may be absent);
  - keep every cam0 manifest row as a frame;
  - set every `cam1` to `None`;
  - reduce the calibration to camera 0;
  - `is_monocular()` reports the mode.
- `vio_estimator_from_calibration(calibration, config)`: the estimator
  construction that `BasaltVioEstimatorAdapter::from_config` already did,
  moved into a public function. `from_config` calls it with the same
  arguments, in the same order.
- `basalt_euroc_vio_demo --mono`:
  - opens the recording with `open_monocular`;
  - appends `camera_mode=mono_inertial` to the summary (only in this mode);
  - is rejected together with `--native-companion-binding`, which binds
    stereo native captures.
  - Every other flag works as before, including `--pipeline`,
    `--no-marg-data` and `--no-trace`.

## Stereo path is unchanged

- `EurocSensorDataset::open` and `open_with_timing` pass `monocular = false`.
  That branch runs the same statements as before.
- No estimator, window, AOM, landmark or frontend arithmetic was edited.
- The demo's stereo branch calls the same constructors. Its summary text
  gets no new line.
- All existing `visloc-basalt` tests, including the trajectory/MargData
  contract tests, pass unchanged (see "Verification").

## How to run

```bash
cargo run --release --example basalt_euroc_vio_demo -- \
  --euroc-dir /data/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/euroc_config.json \
  --out-dir target/basalt_mh01_mono --mono --no-marg-data --no-trace
```

This writes `trajectory.tum` and `trajectory.csv` as in the stereo replay.
Evaluate them with the usual tooling. Monocular VIO is metric, so use SE(3)
alignment. Sim(3) alignment would hide a scale error.

## Verification

Synthetic tests in `pipelines/basalt/tests/mono_inertial_synthetic.rs`:

- **Rig and scene:** an ideal pinhole camera (Double Sphere with
  `xi = alpha = 0`), 752×480, rigidly mounted forward-looking on an IMU.
- **Motion:** smooth 3-D translation (≈5–7 m path) and roll/pitch/yaw
  rotation that starts at rest.
- **IMU:** 200 Hz, with constant biases (accel ≈0.06 m/s², gyro ≈0.0027 rad/s)
  plus white noise.
- **Camera:** 20 Hz.
- **Estimator setup:** built from a one-camera calibration and the pinned
  `configs/basalt/euroc_config.json`, the same way `--mono` builds it
  (f32 upstream scalar mode).

Unoptimized, the estimator runs about 3.6 s per frame, roughly 40× slower
than release. For that reason:

- `monocular_inertial_vio_smoke` (40 tracks, 2 s) runs by default under
  `cargo test` and takes ≈40–50 s unoptimized.
- The full-length scenarios are `#[ignore]`d. Run them with:

```bash
cargo test --release -p visloc-basalt --test mono_inertial_synthetic -- --include-ignored --nocapture
```

They take ≈60 s on two threads and print a one-line summary per scenario.

| Test | Input | Result (release; the smoke result is bit-identical in debug) |
| --- | --- | --- |
| `monocular_inertial_vio_smoke` (default) | Ideal tracks, 40 per frame, 2 s | RMS 15 mm over a 1.6 m path; scale 1.013; dead reckoning 0.14 m |
| `monocular_inertial_vio_tracks_with_metric_scale` | Ideal tracks with 0.5 px noise, KLT-like loss/replenish, lean path | RMS 5 mm, max 21 mm over a 6.8 m path; displacement scale 1.0007; IMU-only dead reckoning drifts 3.0 m |
| `monocular_inertial_margdata_path_emits_valid_mapper_packets` | Same, full `process` path | Every MargData passes `validate_contract`; 8 mapper packets; RMS 9 mm, scale 0.998 |
| `monocular_inertial_adapter_tracks_rendered_images` | **Rendered textured-room images → `DirectKltStream` → estimator through `BasaltVioEstimatorAdapter`**, cam1 = `None` | RMS 7 mm, max 13 mm over a 5.1 m path; scale 1.000; dead reckoning 1.5 m |
| `monocular_inertial_static_start_drifts_then_recovers` | 2 s at rest, then motion | Drift at rest up to 0.13 m (IMU-only; see limitations); ≤ 8 mm from 2 s after motion onset; scale 1.002 |

The first visual factors enter the window at frame 14 (0.7 s). That is when
the IMU-propagated baseline first passes the 5 cm triangulation gate.

Further checks:

- `tests/calibration_contract.rs::retain_cameras_produces_cam0_only_monocular_rig`
- `tests/klt_stream_contract.rs::direct_stream_runs_monocular_without_cam1_or_stereo_stages`
- `euroc::tests::monocular_reader_ignores_cam1_and_keeps_every_cam0_frame`
- the demo's `mono_flag_defaults_off_and_is_parsed`

The test bounds are loose compared with the observed values (RMS < 5 cm,
max < 10 cm, |scale − 1| < 2 %) so that floating-point differences between
platforms do not make them flaky.

## Known limitations / open items

1. **No EuRoC numbers yet.** The ATE of `--mono` on the 11 EuRoC sequences
   (SE(3)-aligned, compared with the stereo-inertial baseline and with
   ORB-SLAM3 mono-inertial) still needs to be measured. Also measure the
   real-time factor, and record `first_visual_frame`-style statistics, i.e.
   when temporal triangulation starts.
2. **Static start.** While the body is at rest, a single camera has no
   parallax. Translation then follows the IMU alone, and the drift is
   driven by accelerometer bias until motion begins. The static-start test
   pins this behavior: about 0.13 m after 2 s at rest with a 0.06 m/s² bias,
   then recovery. EuRoC MH sequences start on the ground, so expect a
   visible error at the start of those runs.
   Upstream Basalt has no zero-velocity update. Adding one, or a
   parallax-gated init in the style of VINS-Mono, would be a non-upstream
   extension to the parity-locked window solver. It is deliberately not part
   of this change.
3. **Constant-velocity segments.** These keep metric scale only through the
   IMU's bias and gravity terms, the textbook mono-VIO observability limit.
   The synthetic tests always contain acceleration.
4. **Mapping / loop closure.** The online SLAM demo and the NFR mapper stay
   stereo-only.
5. **Triangulation gate.** The 5 cm gate is still the hard-coded upstream
   constant (`vio_min_triangulation_dist` in the EuRoC config has the same
   value). Mono-specific tuning, such as a parallax-angle gate, has not been
   explored.
