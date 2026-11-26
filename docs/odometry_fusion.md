# IMU, wheel and legged odometry in map localization

Map localization with `--vio` (and in `visloc_map_localize_node`) already uses
the IMU: the stereo-inertial Basalt VIO gives a pose for every frame, and
`OdomCorrection` keeps a rigid yaw + translation from the VIO frame to the
map, averaged over the recent map fixes. The coupling is loose: the map never
reaches the VIO's own estimate (velocity, biases), and a wheel odometry is an
alternative to the VIO, not a partner. This page tries four ways to couple
more, measured on the same maps and queries:

| | What | Where |
|---|---|---|
| **A** | Each fix the correction accepts goes into the VIO's sliding window as a pose prior (in the VIO frame, through the correction) | `ExternalMeasurement::PosePrior` in `pipelines/basalt/src/vio/window/external.rs`; `--vio-map-prior` |
| **B** | A sliding-window 4-DoF pose graph over the odometries and the fixes replaces the single rigid correction; several odometries are fused | `visloc_localization::MapFusion` (`pipelines/localization/src/fusion.rs`); `--fusion` |
| **C** | The wheel odometry's body velocity goes into the VIO's window on every frame (wheel-inertial odometry) | `ExternalMeasurement::BodyVelocity`; `--vio-wheel-velocity` |
| **D** | A legged robot's kinematic odometry as the odometry, alone or fused with the VIO | `--odometry FILE --odometry-noise legged` (with `--fusion`); `scripts/synthesize_leg_odometry.py` |

A and C are factors added to the VIO's window problem. They are not part of
upstream Basalt. With none queued, the window builds exactly the factors it
always did, and the Basalt port's parity tests pass unchanged. A couples the
map at the pose level (the fix's PnP pose), not landmark by landmark.

## Setup

- **OpenLORIS-Scene office** (wheeled robot, RealSense T265 stereo + IMU,
  wheel odometry, mocap ground truth). Converted as in
  [openloris_map_localization.md](openloris_map_localization.md). Map A from
  office1-1 and map B from office1-2. Queries office1-6 → A (2.8 days later,
  furniture moved) and office1-7 → B (driven the other way, so few fixes).
  Each pair runs twice: with the dataset's calibration ("raw") and with the
  stereo baseline scaled by 0.986 ("scaled"; maps 0.75 / 0.88 cm SLAM ATE
  instead of 2.76 / 1.58 cm). The wheel odometry's `--imu-from-body` is the
  calibrated one from that page.
- **EuRoC MH_02 → MH_01 map** (no wheels). For D, a synthetic legged
  odometry made from MH_02's ground truth by `scripts/synthesize_leg_odometry.py`
  (defaults, seed 0): 3% scale error, 0.5°/m heading drift, a yaw random
  walk, 1 cm/m vertical drift, 2 Hz gait bounce and sway. Alone it is
  1.11 m RMSE after a rigid fit over the 73 m flight. No public dataset pairs
  a legged robot's odometry with a map-localization benchmark, so treat D's
  numbers as a test of the fusion, not of a real robot.
- Localization of every 5th frame (`e5`) or every 30th (`e30`: 1 s apart on
  OpenLORIS, 1.5 s on EuRoC).
  Everything else is the default of `basalt_euroc_localize_demo`.
- Scored in the "fixed" mode of `scripts/evaluate_map_localization.py`:
  through the map's own SLAM-to-ground-truth alignment, so the map's own
  error is included and nothing is fitted to the scored poses. The fraction
  of frames with a map pose is 99.6–99.8% in every OpenLORIS run and 99.97%
  on EuRoC, except where noted.

## Results

RMSE / max of every frame's map pose, in cm.

### OpenLORIS office (wheeled robot)

| Query → map | Baseline (VIO + correction) | A | B | C | **A + C** | B, VIO + wheel | Wheel + correction |
|---|---|---|---|---|---|---|---|
| office1-6 → A, raw, e5 | 3.76 / 4.6 | 3.57 / 4.6 | 3.63 / 4.6 | 3.70 / 4.6 | **3.42** / 4.7 | 5.42 / 7.5 | 7.38 / 11.3 |
| office1-6 → A, raw, e30 | 3.67 / 4.5 | 3.48 / 4.5 | 3.70 / 4.7 | 3.76 / 5.0 | **3.34** / 4.5 | 5.75 / 7.5 | 7.42 / 11.2 |
| office1-6 → A, scaled, e5 | 1.88 / 2.9 | 1.52 / 2.4 | 1.79 / 2.7 | 1.85 / 2.8 | **1.46** / 2.2 | 4.20 / 7.7 | 5.34 / 10.1 |
| office1-6 → A, scaled, e30 | 1.93 / 2.8 | 1.71 / 2.5 | 2.14 / 3.6 | 1.88 / 2.7 | **1.66** / 2.4 | 4.66 / 8.6 | 5.52 / 10.3 |
| office1-7 → B, raw, e5 | 6.62 / 13.4 | 6.83 / 13.9 | 6.69 / 13.5 | 5.89 / 12.0 | 6.08 / 12.5 | **4.13** / 7.7 | 5.19 / 10.9 |
| office1-7 → B, raw, e30 | 7.04 / 14.1 | 6.96 / 14.0 | 7.21 / 14.4 | 6.30 / 12.8 | 6.22 / 12.6 | **4.70** / 8.5 | 5.23 / 10.9 |
| office1-7 → B, scaled, e5 | 5.60 / 11.1 | 5.00 / 10.2 | 5.66 / 11.2 | 5.18 / 10.4 | **4.56** / 9.5 | 4.66 / 7.7 | 4.91 / 9.1 |
| office1-7 → B, scaled, e30 | 5.35 / 10.8 | 5.35 / 10.8 | 5.49 / 11.0 | 4.92 / 10.0 | 4.91 / 10.0 | **4.59** / 7.7 | 5.08 / 9.4 |

### EuRoC MH_02 → MH_01 (drone; synthetic legged odometry for D)

| | e5 | e30 |
|---|---|---|
| Baseline (VIO + correction) | 2.02 / 5.7 | 2.05 / 6.0 |
| A | 1.89 / 6.7 | **1.93** / 5.5 |
| B | 1.98 / 5.9 | 2.33 / 6.4 |
| A + B | **1.79** / 6.7 | 1.98 / 5.6 |
| D: legged odometry + correction | 5.50 / 12.4 (98.6%) | **5.97** / 15.5 (98.6%) |
| D: legged odometry + B | **2.73** / 6.4 (98.6%) | 6.27 / 17.4 (98.6%) |
| D: VIO + legged, B | 1.94 / 5.4 | 2.50 / 6.4 |
| VIO cut out 40–70 s, B, VIO only | 1.92 / 5.9 (**80.2%**) | 2.18 / 6.4 (**79.9%**) |
| VIO cut out 40–70 s, B, VIO + legged | 2.08 / 5.3 (**99.97%**) | 3.89 / 15.8 (**99.97%**) |

## What this shows

- **A + C is the best VIO setup for a wheeled robot.** It beats the baseline
  in all eight OpenLORIS runs, by 0.3 to 1.0 cm (8–22%). The two help in
  different places. A helps where fixes keep coming (office1-6): the map
  pulls the VIO's velocity and biases, not only its output. C helps where
  fixes are rare (office1-7): the wheels hold the VIO's speed between them.
  On the scaled maps, office1-6 drops from 1.88 to 1.46 cm.
- **A alone also helps without wheels.** On EuRoC it gives 1.89 / 1.93 cm
  against 2.02 / 2.05 cm, and it is better or equal in 7 of the 8 OpenLORIS
  runs. The exception is office1-7 raw e5, 6.83 against 6.62 cm. There the
  map's scale error makes the fixes disagree with the VIO.
- **B does not replace the rigid correction for one odometry.** With the
  VIO alone it is within 0.15 cm of the baseline at e5 and worse at e30 (2.33 against
  2.05 cm on EuRoC). Between rare fixes, a VIO that hardly drifts is better
  held rigid than bent toward each fix.
- **B is what keeps a pose when the VIO fails.** With the VIO cut out for
  30 s, the VIO-only fusion has no map pose for a fifth of the flight (it
  starts over at the next fix after the outage rather than guess). Fused
  with the legged odometry, 99.97% of the frames keep a map pose, at
  2.1 cm with fixes every 5th frame. With fixes every 1.5 s it is
  3.9 cm, 15.8 cm worst.
- **B with VIO + wheels pays off only where the VIO is weak.** On office1-7
  raw it gives 4.13 cm against 6.62 cm. On office1-6 it is worse (4.2 against
  1.9 cm scaled). The wheel odometry's own error (5–7 cm against the map,
  slip in turns in place) pulls a good VIO off. A + C uses the wheels'
  speed only, inside the VIO, and does not have this problem.
- **Legged odometry (D) needs fixes to be frequent for B to win.** With
  fixes every 5th frame, B halves the rigid correction's error (2.73
  against 5.50 cm): a leg odometry drifts within seconds, and the graph
  bends it between fixes. With fixes 1.5 s apart, both are near 6 cm, and B is
  slightly worse. Fused with a VIO, the legged odometry costs little where
  the VIO works (1.94 against 2.02 cm) and carries the pose where it does
  not.

## Monocular VIO

One camera and an IMU, with or without wheels.

The same couplings with the monocular-inertial VIO (`--mono`: cam0 + IMU,
[mono_inertial_vio.md](mono_inertial_vio.md)), every 5th frame localized,
the scaled calibration, and
`configs/basalt/variants/official_euroc_ds/euroc_config_mono_static_start.json`
(the EuRoC config with `config.vio_init_accel_average_s: 0.5`, as the robots
start still). Two kinds of map: the stereo SLAM maps above, and
maps built by the monocular SLAM itself (`basalt_euroc_online_slam_demo
--mono --save-map`; office1-1 1.37 cm and office1-2 1.80 cm SLAM ATE, against
0.75 / 0.88 cm for stereo). "Mono map" is a robot with one camera
throughout. RMSE / max in cm.

| Query → map | VIO + correction | A | C | **A + C** | Stereo A + C, for reference |
|---|---|---|---|---|---|
| office1-6 → stereo map | 24.76 / 41.6 | 8.62 / 16.4 | 3.80 / 8.8 | **1.72** / 2.7 | 1.46 / 2.2 |
| office1-7 → stereo map | 50.45 / 107.9 | 13.40 / 28.7 | 4.66 / 12.6 | **4.58** / 12.7 | 4.56 / 9.5 |
| office1-6 → mono map | 7.66 / 15.7 | 2.71 / 8.8 | 2.35 / 6.7 | **1.40** / 3.6 | |
| office1-7 → mono map | 49.75 / 106.5 | 30.64 / 56.3 | **4.62** / 12.3 | 5.82 / 11.5 | |
| EuRoC MH_02 → stereo MH_01 map | 5.23 / 12.5 | **2.07** / 5.8 | — | — | 1.79 / 6.7 (A + B) |

Frames with a map pose: 98.2–99.6%.

- **A monocular VIO alone does not hold on a ground robot.** Corrected
  into the map it is 8–50 cm off; alone (rigid fit) 16.5 cm on office1-6
  and 38.5 cm on office1-7. A robot driving at near constant speed on a
  level floor barely excites the IMU, so the monocular scale and velocity
  drift.
- **The wheels fix that.** With the wheel velocity in the VIO (C), the
  monocular VIO alone is 3.4 cm on office1-7, and A + C reaches the stereo
  numbers: 1.72 against 1.46 cm, and 4.58 against 4.56 cm, on the same maps.
  On a map built by the monocular SLAM, office1-6 is 1.40 cm. With rare
  fixes (office1-7), C alone is as good or better than A + C.
- **Without wheels, the map fixes alone (A) do most of the job where fixes
  keep coming.** On EuRoC, A brings the monocular VIO from 5.23 to 2.07 cm,
  the stereo baseline's 2.02 cm. On office1-7, where fixes are rare, A
  alone is not enough (13–31 cm).
- **No roll/pitch prior for a monocular VIO without wheels.** With the
  fix's roll and pitch in the prior (1 deg), the monocular VIO on office1-7
  ran away (49 m RMSE): the map's gravity is a degree or so off the IMU's,
  and pulling the tilt leaks gravity into the velocity. Without the tilt
  rows: 13.4 cm. With wheels or stereo, the tilt prior helps a little
  (office1-6/7 stereo A + C: 1.46 / 4.56 cm with it, 1.56 / 5.18 cm
  without), so it stays on there. `--map-prior-sigma` and the ROS 2 node's
  `map_prior_tilt_stddev_deg` default to no tilt for a monocular VIO without
  wheel velocity.
- **Through the ROS 2 nodes** (`mono:=true`, mono maps, replayed over DDS at
  0.2x): office1-6 7.87 → 1.30 cm with `map_prior` + `wheel_velocity`.
  Before the node took the averaged initial tilt and the config's LM
  iterations with `mono`, office1-6 was 57.8 → 1.88 cm and office1-7
  60.3 → 7.89 cm
  ([ROS 2 README](../ros2/visloc-ros2/README.md#map-fixes-and-wheel-speed-inside-the-vio)).

## Monocular without wheels: a non-holonomic constraint (not adopted)

A robot without wheel odometry still moves like one: its body goes along its
own x axis. On office1-6, the ground truth taken to the wheel axle
(`--imu-from-body`) moves forward at 0.21 m/s at the median while moving,
sideways and vertically at 0.5 and 0.8 cm/s (90th percentile 1.2 and
1.4 cm/s). So a zero lateral and vertical body velocity was put into the
monocular VIO's window on every frame, through the same body-velocity factor
as C but with the forward row weighted 0 (commit `b3ae377`,
`--vio-nonholonomic SIGMA`, since removed). RMSE in cm, mono maps; the last
columns are the VIO alone, aligned by a similarity (scale removed), and the
scale that alignment needs:

| | after map correction | VIO, scale removed | scale |
|---|---:|---:|---:|
| office1-6, VIO + correction | 7.66 | 6.00 | 1.109 |
| — constraint 0.3 m/s | 12.24 | 6.46 | 1.188 |
| — constraint 0.1 m/s | 13.98 | 5.56 | 1.241 |
| — constraint 0.03 m/s | 9.42 | 4.35 | 1.144 |
| office1-6, A | 2.71 | | |
| — A + constraint 0.1 m/s | 2.65 | | |
| office1-7, VIO + correction | 49.75 | 11.55 | 1.294 |
| — constraint 0.1 m/s | 46.50 | 37.81 | 1.095 |
| — constraint 0.03 m/s | 21.89 | 17.12 | 1.029 |
| office1-7, A | 30.64 | 4.95 | 1.165 |
| — A + constraint 0.1 m/s | 82.27 | 65.64 | 1.210 |
| — A + constraint 0.03 m/s | 138.05 | 87.38 | 1.798 |

- **The constraint holds, but it does not observe the scale**, which is most
  of a monocular VIO's error on a ground robot. On office1-6 it straightens
  the shape (6.00 to 4.35 cm with the scale removed) while the scale gets
  worse, and the map-corrected error grows.
- **It is not stable.** office1-7 gains at 0.03 m/s and barely at 0.1 m/s,
  and with the map prior (A) it runs away at both.
- Without wheels, the scale comes from the map fixes (A). The wheels (C)
  remain the way to hold a monocular VIO between fixes.

A scale in the map correction does not help either. The correction keeps a
yaw and a translation, averaged over the fixes of the last 5 s. A causal
re-run of it from the saved VIO trajectory and accepted fixes (scripted, not
in the repository) reproduces office1-6's 7.66 cm. Adding a scale fitted
over the fixes of the last 10, 20 or 30 s gives 8.99, 9.66 and 9.20 cm: the
scale (about 1.15) is real, but where fixes keep coming, the 5 s window
re-anchors the trajectory before the scale error builds up. office1-7 is
where a scale would matter, and there it has nothing to fit: every run
against office1-2's map gets its fixes in the first 2 s of 37 s (stereo map
and VIO included; one monocular run got 26 fixes up to 10.5 s). The rest of
the walk leaves what the map covers. Its 13 to 50 cm come from that, and
more of the map, or images that still localize there
([openloris_map_localization.md](openloris_map_localization.md)), is what
would help.

## What was adopted

- **A and C** stay as options of `basalt_euroc_localize_demo`
  (`--vio-map-prior`, `--vio-wheel-velocity`), with the factor types public
  on the Basalt estimator (`BasaltVioEstimator::add_external_factor`).
  `--vio-map-prior --vio-wheel-velocity` is the recommended setup for a
  wheeled robot with a VIO. They are opt-in, so the Basalt port's default
  output is unchanged. On ROS 2 they are `visloc_vio_node`'s `map_prior`
  (fed by `visloc_map_localize_node`'s new `odom_fix` topic) and
  `wheel_velocity`. Replayed over DDS, they lower office1-6's error from
  2.25 to 1.66 cm and office1-7's from 6.38 to 4.50 cm
  ([ROS 2 README](../ros2/visloc-ros2/README.md#map-fixes-and-wheel-speed-inside-the-vio)).
- **B** (`MapFusion`, `--fusion`) is for robots with more than one odometry,
  to keep a map pose through a VIO outage. It is not the default.
- **D** needs no new input. Any gravity-aligned 6-DoF odometry (TUM file, or
  `nav_msgs/Odometry` for `visloc_map_localize_node`) already works.
  `OdometryNoise::LEGGED` is the fusion's drift model for one, and
  `scripts/synthesize_leg_odometry.py` makes a test odometry from ground
  truth. The wheel calibration (`WheelCalibration`) is planar and does not
  apply to legged robots.

## Reproduce

```bash
# Maps as in openloris_map_localization.md (office1-1 -> o11, office1-2 -> o12),
# or with calibration_scaled.json (scripts/scale_stereo_baseline.py --scale 0.986).
D=data; C=configs/basalt/variants/official_euroc_ds/euroc_config.json
IFB=-0.0314,-0.8938,-0.2587,-0.5059,-0.5042,-0.4958,0.4940

# A + C on office1-6 against map A
target/release/examples/basalt_euroc_localize_demo --map runs/o11_map \
  --euroc-dir $D/office1-6 --out-dir runs/o16_ac --vio \
  --calibration $D/office1-6/calibration.json --config $C --localize-every 5 \
  --odometry $D/office1-6/odom.tum --imu-from-body $IFB \
  --vio-map-prior --vio-wheel-velocity

# B, VIO + wheels
target/release/examples/basalt_euroc_localize_demo --map runs/o12_map \
  --euroc-dir $D/office1-7 --out-dir runs/o17_fusion --vio \
  --calibration $D/office1-7/calibration.json --config $C --localize-every 5 \
  --odometry $D/office1-7/odom.tum --imu-from-body $IFB --fusion

# D: a synthetic legged odometry for MH_02, fused with the VIO through a
# 30 s VIO outage
python3 scripts/synthesize_leg_odometry.py \
  --ground-truth $D/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv \
  --out $D/MH_02_easy/leg_odom.tum
target/release/examples/basalt_euroc_localize_demo --map runs/mh01_map \
  --euroc-dir $D/MH_02_easy --out-dir runs/mh02_leg --vio \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config $C --localize-every 5 \
  --odometry $D/MH_02_easy/leg_odom.tum --odometry-noise legged --fusion \
  --drop-vio-s 40:70

# Score (fixed mode: the map's own alignment to ground truth)
python3 scripts/evaluate_map_localization.py --mode fixed \
  --estimate runs/o16_ac/trajectory_map.tum \
  --reference $D/office1-6/mav0/state_groundtruth_estimate0/data.csv \
  --align-estimate runs/o11_slam/trajectory_online.tum \
  --align-ground-truth $D/office1-1/mav0/state_groundtruth_estimate0/data.csv

# Monocular, on a map from the monocular SLAM, with the wheels
target/release/examples/basalt_euroc_online_slam_demo --euroc-dir $D/office1-1 \
  --calibration $D/office1-1/calibration_scaled.json --mono \
  --config configs/basalt/variants/official_euroc_ds/euroc_config_mono_static_start.json \
  --out-dir runs/mo11_slam --save-map runs/mo11_map
target/release/examples/basalt_euroc_localize_demo --map runs/mo11_map \
  --euroc-dir $D/office1-6 --out-dir runs/mono16_ac --vio --mono \
  --calibration $D/office1-6/calibration_scaled.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config_mono_static_start.json \
  --localize-every 5 --odometry $D/office1-6/odom.tum --imu-from-body $IFB \
  --vio-map-prior --vio-wheel-velocity
```

## Limits

- One office scene and one EuRoC pair. The office paths are short (6 m),
  and the map's own error is a large part of every number.
- The legged odometry is synthetic.
- A's pose priors leave with their frame. They are not carried into the
  VIO's marginal prior.
- B's drift models (`OdometryNoise::VIO`, `WHEEL`, `LEGGED`) were set once
  and not tuned on these runs.
- The monocular VIO without wheels depends on its start: office1-6 on the
  mono map is 7.7 cm offline with the averaged initial tilt and diverges
  (199 m) from one IMU sample. The ROS 2 node now starts the same way
  (7.87 cm). The wheel-coupled numbers are the stable ones.
