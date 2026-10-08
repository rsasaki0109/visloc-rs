# LaMAria Basalt variants

Per-sequence calibration/config variants for the Project Aria LaMAria benchmark
(see [`docs/lamaria_plan.md`](../../../docs/lamaria_plan.md) and
[`docs/lamaria_stage0.md`](../../../docs/lamaria_stage0.md)).

## Calibrations

- `<seq>_calib.json`: LaMAria's shipped Kalibr IMU noise, converted from the
  pinhole camera model to Basalt's Double Sphere with `xi = 0`, `alpha = 0`.
- `<seq>_calib_variantA_default_noise.json`: the same cameras, but Basalt's
  own EuRoC-default IMU noise. **Use this one** — on R_01_easy the datasheet
  noise gives SE3 ATE 1.60 m vs 0.43 m for variant A.

## VIO configs

- `euroc_config.json`: EuRoC-derived Basalt VIO config (default sliding window:
  `vio_max_states = 3`, `vio_max_kfs = 7`).
- `euroc_config_mapper_reduced_points.json`: the same config with
  `mapper_detection_num_points` 800 → 200, used for the offline mapper runs.
- `euroc_config_big_window.json`: **the larger sliding window found to matter
  at km scale** — `vio_max_states = 10`, `vio_max_kfs = 30`, everything else
  unchanged.
- `euroc_config_big_window_multicam.json`: the big window plus the opt-in
  multi-camera keys for Aria's divergent camera pair (cameras ~75° apart):
  `optical_flow_matching_guess_type = "REPROJ_FIX_DEPTH"` with
  `optical_flow_matching_default_depth = 2.0` (stereo seed from the
  extrinsics instead of the same pixel), `optical_flow_detect_all_cameras`
  (FAST replenishment in cam1 too), `vio_landmarks_all_cameras` (cam1-hosted
  landmarks) and `vio_kf_connectivity_all_cameras`. **Not yet measured on
  LaMAria**; see [`docs/lamaria_multicam.md`](../../../docs/lamaria_multicam.md)
  for the diagnosis, synthetic results and the A/B commands. The detection
  grid is unchanged (50 px; 15×11 cells on 758×572).

## Why the big window

Basalt's default window is tuned for short, room-scale sequences. With it,
LaMAria VIO-only scores **49.65** on `R_11_5cp` and **12.75** on the 1.5 km
`sequence_1_19` (18,352 frames). Enlarging the window is a config-only,
parity-safe change:

| sequence (frames) | metric | default 3/7 | **10/30** |
|---|---|---:|---:|
| R_11_5cp (9.5k) | CP Score | 49.65 (CP@1m 20 %) | **63.19** (CP@1m 60 %) |
| R_11_5cp | pGT R@1m / R@5m | 16 % / 100 % | **51.8 %** / 100 % |
| sequence_1_19 (18k) | CP Score | 12.75 (CP@1m 7.1 %) | **27.09** (CP@1m 14.3 %) |
| sequence_1_19 | pGT R@1m / R@5m | 0 % / 5.4 % | 4.6 % / **30.0 %** |
| R_12_10cp (20k) | CP Score | **30.00** (CP@1m 0 %) | 28.55 (CP@1m 10 %) |
| R_12_10cp | pGT R@1m / R@5m | 4.4 % / **66.9 %** | **7.6 %** / 62.7 % |

The window helps strongly on `R_11_5cp` and `sequence_1_19` but is **neutral to
slightly negative on `R_12_10cp`** (mean CP Score 30.80 → 39.61 over the three),
and the per-sequence spread is dominated by the 5–15-control-point piecewise
Score's high variance: a single control point moving from <5 cm to >10 m swings
the Score by ~14 points, so configs must be ranked over several sequences (or
with CP@1m / pGT recall alongside) rather than one. On `R_11_5cp` the
larger-window VIO alone beats the default-window offline mapper (57.66) and
approaches microSLAM (mono, 64.5); the best published academic baseline is 27.7.

Global ATE keeps improving further at `vio_max_states = 15` / `vio_max_kfs =
45` (sequence_1_19 SE3 13.60 → 11.91 m) but the official 14-control-point
Score drops to 22.27, so the window should be selected on the official metric,
not ATE. Wall time for the 18,352-frame sequence: 12 min (3/7), 58 min (10/30),
1 h 51 min (15/45). Note the marginalization packets at 10/30 are ~3.7× larger
(~73.5 MB vs ~20 MB), so the file-based offline mapper dump is impractical at
this window size.
