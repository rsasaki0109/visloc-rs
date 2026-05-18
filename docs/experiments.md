# Experiments

Initial experiments should stay focused on map-based localization:

- Synthetic 2D-3D correspondences with known pose
- COLMAP text model loading smoke tests
- Descriptor matching quality with ratio-test thresholds
- PnP RANSAC sensitivity to reprojection threshold and outlier ratio
- IO-backed localization with `cargo run --example localize_colmap_text`
- Dependency-free grayscale corner extraction with `cargo run --example localize_with_corner_extractor`
- PGM-backed grayscale image localization with `cargo run --example localize_from_pgm`
- Optional PNG/JPEG-backed grayscale image localization with `cargo run --features image-io --example localize_from_common_image`
- Optional PNG/JPEG-backed image sequence tracking with `cargo run --features image-io --example track_image_sequence_from_common_images`
- Timestamped PNG/JPEG-backed image sequence tracking with GNSS priors using `cargo run --features image-io --example track_timestamped_image_sequence_with_gnss_prior`
- `scripts/check_timestamped_gnss_image_demo_outputs.sh` verifies the timestamped image GNSS-prior demo images, timestamp file, GNSS log, and sync evaluation JSON.
- The timestamped image GNSS-prior demo output guide is in [timestamped_gnss_image_demo.md](timestamped_gnss_image_demo.md).
- KITTI-style image sequence loading with `cargo run --features image-io --example load_kitti_image_sequence`
- `scripts/check_kitti_image_sequence_demo_outputs.sh` verifies the KITTI-style image sequence demo images, timestamp file, calibration file, and loader output log.
- The KITTI-style image sequence demo output guide is in [kitti_image_sequence_demo.md](kitti_image_sequence_demo.md).
- File-based sequence localization and tracking report export with `cargo run --example localize_sequence_from_files -- --out-dir target/visloc_sequence_demo`
- Localization-based tracking state transitions with `cargo run --example track_sequence_dummy`
- Tracking report export with `cargo run --example track_sequence_dummy -- --out-dir target/visloc_tracking_demo`
- Trajectory evaluation with `cargo run --example evaluate_trajectory_dummy`
- KITTI file-based trajectory evaluation with `cargo run --example evaluate_trajectory_from_kitti_files -- --out-dir target/visloc_eval_kitti`
- KITTI odometry leaderboard-style relative-motion evaluation with `cargo run --example evaluate_kitti_odometry_benchmark -- --out-dir target/visloc_eval_kitti_odometry`
- KITTI deep stereo VO training-sequence triage with `scripts/run_kitti_deep_vo_train_benchmark.sh --max-frames 260`, which runs the seq smoke path over 00-10 and writes a consolidated `summary.csv` / `summary.md`.
- TUM file-based trajectory evaluation with `cargo run --example evaluate_trajectory_from_tum_files -- --out-dir target/visloc_eval`
- KITTI / TUM ATE trajectory evaluators support pass/fail thresholds with `--max-mean`, `--max-rmse`, `--max-max`, `--min-matched`, and `--min-match-ratio`; `scripts/check_trajectory_evaluation.sh` runs those fixture checks plus the KITTI odometry `t_rel` / `r_rel` fixture.
- Browser-viewable reports are written as `trajectory_report.html` / `tracking_report.html`, with frame-level tracking diagnostics in `tracking.csv` and aggregate tracking metrics in `tracking_summary.json`
- Moving-camera GNSS-prior submap narrowing with `cargo run --example track_sequence_with_gnss_prior -- --out-dir target/visloc_gnss_tracking_demo`, including an `index.html` dashboard, `manifest.json`, tracking diagnostics, `tracking_evaluation.json`, KITTI/TUM poses, synthetic-reference translation errors, and trajectory CSV / JSON / HTML exports
- The GNSS-prior demo output guide is in [gnss_demo.md](gnss_demo.md).
- CI checks the moving-camera GNSS dashboard demo, timestamped image GNSS-prior demo, and KITTI-style image sequence demo; it uploads the checked output directories as `gnss-demo-outputs`, `timestamped-gnss-image-demo-outputs`, and `kitti-image-sequence-demo-outputs` artifacts.

Future experiments can add image feature extraction, online Visual SLAM, inertial priors, and public automotive or UAV sequence data after the visual localization slice is stable.

## KITTI Deep Stereo VO Threshold Sweep

KITTI 00 seq00 local open-data subset, stride 1, 260 frames, deep frontend
(`HogLikeFeatureExtractor` + `MutualSoftmaxMatcher`), 1000 relative-pose RANSAC
iterations, no stereo BA. These are local diagnostics against public seq00
ground truth, not held-out KITTI leaderboard submissions.

This sweep was run before the guarded PnP refinement below.

| PnP reprojection gate | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3.310 px | 2.506 / 2.607 / 3.783 m | **0.9809%** | 0.01436 deg/m | 2.9035% | Reject: rotation regresses |
| 3.315 px | **2.502 / 2.602 / 3.772 m** | **0.9755%** | 0.01438 deg/m | 2.8994% | Reject: best translation, worse rotation |
| 3.318 px | 2.510 / 2.611 / 3.779 m | 0.9866% | **0.01417 deg/m** | 2.8990% | Reject: translation regresses |
| 3.320 px | 2.509 / 2.610 / 3.777 m | 0.9856% | 0.01419 deg/m | **2.8988%** | Adopt: best balanced gate |

The adopted default stays at `3.32 px`: it improves both `t_rel` and `r_rel`
over the earlier `3.35 px` default while keeping the worst 100 m translational
window below the nearby alternatives.

## KITTI Deep Stereo VO Feature Cap Sweep

Same local KITTI 00 seq00 260-frame diagnostic as above, with the adopted
`3.32 px` PnP reprojection gate and 1000 relative-pose RANSAC iterations.

| `deep_max_features` | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1500 | 2.509 / 2.610 / 3.777 m | **0.9856%** | **0.01419 deg/m** | **2.8988%** | Baseline before guarded PnP refinement |
| 1300 | **2.283 / 2.350 / 3.134 m** | 1.2233% | 0.01751 deg/m | **2.5411%** | Reject: ATE and worst translation improve, but mean KITTI windows and rotation regress; one Kabsch fallback |
| 1400 | Incomplete | Incomplete | Incomplete | Incomplete | Reject: pathological runtime, stopped after more than 24 minutes |
| 1800 | 2.548 / 2.648 / 3.811 m | 1.2803% | 0.01630 deg/m | 2.9231% | Reject: more features add ambiguity and regress KITTI windows |

The next performance work should target match quality or pose scoring rather
than simply raising the feature cap.

## KITTI Deep Stereo VO Pose Scoring Trials

Same local KITTI 00 seq00 260-frame diagnostic, using the adopted
`deep_max_features=1500`, `3.32 px` PnP gate, and 1000 relative-pose RANSAC
iterations.

| Trial | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline PROSAC sampling, unweighted inlier-count scoring | **2.509 / 2.610 / 3.777 m** | **0.9856%** | **0.01419 deg/m** | **2.8988%** | Adopted before guarded PnP refinement |
| Add deep-confidence weighted tie-break to RANSAC candidate scoring | 2.510 / 2.611 / 3.786 m | 0.9877% | 0.01420 deg/m | 2.8988% | Reject: tiny but consistent mean KITTI regression |

## KITTI Deep Stereo VO PnP Refinement Guard

Same local KITTI 00 seq00 260-frame diagnostic and adopted frontend settings.
The guard keeps the Gauss-Newton-refined PnP pose only when it preserves or
improves the RANSAC consensus score; otherwise it falls back to the
least-squares PnP pose estimated from the best inlier set.

| Trial | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Unguarded PnP refinement | 2.509 / 2.610 / 3.777 m | 0.9856% | 0.01419 deg/m | 2.8988% | Replaced |
| Guarded PnP refinement | **2.475 / 2.574 / 3.745 m** | **0.8965%** | **0.01246 deg/m** | **2.8446%** | Adopted before the runtime-stability pass: improves ATE, translation, rotation, and worst window |

After adopting the guard, a small PnP gate spot-check kept the default at
`3.32 px`. Tighter gates can improve ATE but regress KITTI windows, while
looser nearby gates lose the adopted mean translation advantage.

| Guarded PnP gate | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3.20 px | **2.401 / 2.489 / 3.399 m** | 1.0180% | 0.01515 deg/m | 2.8627% | Reject: ATE improves, but KITTI mean translation/rotation regress |
| 3.32 px | 2.475 / 2.574 / 3.745 m | **0.8965%** | **0.01246 deg/m** | **2.8446%** | Adopted gate before the runtime-stability pass |
| 3.34 px | 2.477 / 2.576 / 3.733 m | 0.9327% | 0.01269 deg/m | 2.8511% | Reject: close ATE, worse KITTI means |
| 3.45 px | 2.502 / 2.600 / 3.762 m | 0.9305% | 0.01331 deg/m | 2.8858% | Reject: ATE and KITTI windows regress |

## KITTI Deep Stereo VO Depth / Stereo Refinement Trials

Same local KITTI 00 seq00 260-frame diagnostic and guarded PnP refinement.
These trials tested whether reducing noisy far-field stereo points in PnP, or
turning on the current-frame stereo translation refinement, improves the
leaderboard-style balanced metric.

| Trial | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| No PnP depth cap, no stereo translation refinement | 2.475 / 2.574 / 3.745 m | 0.8965% | **0.01246 deg/m** | 2.8446% | Pre-runtime-fix balanced baseline |
| `pnp_max_depth=60 m` | 2.309 / 2.370 / **3.204 m** | 0.8869% | 0.01827 deg/m | 2.5877% | Reject for balanced default: translation/ATE improve, rotation regresses hard |
| `pnp_max_depth=70 m` | **2.189 / 2.270 / 3.300 m** | **0.8522%** | 0.01866 deg/m | **2.5836%** | Reject for balanced default: best ATE and translation, but rotation regresses hard |
| `pnp_max_depth=75 m` | **2.189 / 2.270 / 3.300 m** | **0.8522%** | 0.01866 deg/m | **2.5836%** | Same as 70 m on this subset; reject for rotation |
| `pnp_max_depth=70 m`, `pnp_reprojection_threshold=2.8 px` | 2.346 / 2.451 / 3.644 m | 1.0719% | 0.01818 deg/m | 2.6114% | Reject: tighter gate loses translation gain and keeps rotation regression |
| `pnp_depth_hypotheses=60 m` guarded selector | **2.002 / 2.065 / 2.666 m** | 0.9306% | 0.01708 deg/m | **2.4395%** | Reject for balanced default: best ATE/max and worst translation, but mean KITTI translation and rotation regress |
| `pnp_depth_hypotheses=70 m` guarded selector | 2.201 / 2.278 / 3.218 m | 0.8863% | 0.01538 deg/m | 2.5908% | Reject for balanced default: improves ATE and `t_rel`, but rotation regresses |
| Stereo translation refinement enabled | 3.044 / 3.176 / 4.520 m | 1.6985% | **0.01246 deg/m** | 3.5649% | Reject: preserves rotation but badly worsens translation drift |

Depth-capped PnP is useful as a translation/ATE ablation, and the guarded
`--pnp-depth-hypotheses` selector can generate those candidates without
changing the default path. The balanced KITTI-style default remains uncapped:
the 70 m fixed cap trades a 0.0443 pp `t_rel` gain for a much larger
0.00619 deg/m `r_rel` loss, and the selector variants still lose too much
rotation.

## KITTI Deep Stereo VO Runtime / Longer-Window Stability

Same local KITTI 00 seq00 diagnostic and adopted frontend settings. The
full 1500-feature / 1000-iteration run previously hit a pathological runtime
around original frames 271→272 because the default PnP path still computed
the Kabsch fallback eagerly. The current frontend stops high-consensus PnP
searches early and only runs Kabsch when it is actually needed as a fallback.

| Trial | Result | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Original frame 271→272 probe | Completed | n/a | n/a | n/a | Full 1500 features / 1000 iterations; pair processed in 0.5 s with 580 PnP inliers, 0.0136 m translation-magnitude error, 0.0947 deg rotation error |
| 260-frame smoke | Completed | 2.494 / 2.631 / 4.185 m | **0.6750%** | 0.01458 deg/m | Adopted runtime-stable default; all 259 pairs sourced from PnP |
| 300-frame smoke | Completed | 2.787 / 2.989 / 5.114 m | 1.2714% | **0.01401 deg/m** | All 299 pairs sourced from PnP; confirms the former 271→272 runtime cliff is gone |
| 900-frame smoke | Completed | 9.587 / 11.917 / 27.583 m | 2.4255% | 0.01666 deg/m | GT length 639.75 m, so public windows reach 600 m but not 700/800 m; 895 PnP pairs and 4 Kabsch fallbacks |
| 900-frame smoke, adaptive 60 m depth rescue | Completed | 9.449 / 11.737 / 26.878 m | 2.3232% | 0.01584 deg/m | Adopted default: preserves the 260-frame result while removing all four 900-frame Kabsch fallbacks |
| 900-frame smoke, `pnp_depth_hypotheses=60 m` | Completed | **7.988 / 10.146 / 24.206 m** | **2.1462%** | **0.01404 deg/m** | Removes all four Kabsch fallbacks and improves ATE, public-length means, 100 m means, and worst windows |

The 900-frame run localizes the main long-window degradation around original
frames 624→654. The largest pair translation-magnitude error is frame 624→625
(`0.358 m`), and the only Kabsch fallbacks occur at 625→626, 628→629,
632→633, and 653→654. The next improvement target is therefore not runtime but
recovering PnP or suppressing low-quality motion updates in that segment.
On a focused 620→674 probe, `pnp_depth_hypotheses=60 m` removes all fallbacks
and reduces mean/max per-pair translation-magnitude error from
`0.0527 / 0.3578 m` to `0.0382 / 0.1606 m`. The adopted default now applies a
more conservative adaptive variant: it tries a 60 m candidate only when the
primary PnP inlier ratio falls below `0.65` or primary PnP fails. This keeps
the 260-frame README result unchanged (`t_rel = 0.6750%`,
`r_rel = 0.01458 deg/m`) while removing the four 900-frame fallbacks. The
manual always-on `--pnp-depth-hypotheses 60` remains useful for long-sequence
diagnostics, but it is not the global default because on the 260-frame subset
it improves ATE (`2.237 / 2.304 / 3.029 m`) while worsening the 100 m KITTI
means (`t_rel = 0.9258%`, `r_rel = 0.01734 deg/m`) relative to the adaptive
default.

## KITTI Deep Stereo VO Training Sequence Triage

The cross-sequence smoke runner was used to avoid tuning only on seq00. The
first failure was the 260-frame seq01 highway slice: GT path length is
537.72 m, but the raw VO path collapses to 420.51 m and public-length
translation error rises to `20.2301%`. The worst per-pair errors occur late
in the slice where true frame-to-frame motion is about 2.7 m but the estimated
translation sometimes falls below 0.1 m, especially under weak PnP consensus.

| Trial | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | Source counts | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| seq00 260-frame default | 2.494 / 2.631 / 4.185 m | **0.6750%** | 0.01458 deg/m | 259 PnP | Guardrail: unchanged by the rescue |
| seq01 260-frame baseline | 27.462 / 46.407 / 131.736 m | 20.2301% | 0.03129 deg/m | 216 PnP / 43 Kabsch fallback | Fails high-speed scale; not leaderboard-competitive |
| seq01 `pnp_depth_hypotheses=60 m` | 28.159 / 46.970 / 132.918 m | 20.2447% | 0.03136 deg/m | n/a | Reject: depth cap does not address this failure mode |
| seq01 `--stereo-pose-refinement` | 25.460 / 42.531 / 120.038 m | 18.6759% | 0.03129 deg/m | n/a | Helps, but remains too high |
| seq01 collapsed-only motion-scale rescue | 26.442 / 44.434 / 123.847 m | 19.3833% | 0.03129 deg/m | n/a | Fixes only the most collapsed translations |
| seq01 collapsed-only rescue + automatic fast/weak stereo refinement | 23.617 / 38.530 / 102.128 m | 16.9531% | 0.03129 deg/m | 216 PnP / 43 Kabsch fallback | Gets most of the explicit-refinement gain without changing seq00 |
| seq01 motion-scale band rescue + automatic fast/weak stereo refinement | 18.563 / 28.603 / 74.120 m | 12.6562% | 0.03129 deg/m | 216 PnP / 43 Kabsch fallback | Rescales weak fast pairs outside the 0.65x-1.6x recent-median band |
| seq01 band rescue + rotation-spike rescue | 19.280 / 28.752 / **72.939 m** | **12.3334%** | **0.02025 deg/m** | 216 PnP / 43 Kabsch fallback | Adopt default: improves KITTI translation/rotation means and max rotation; ATE mean is slightly worse |
| seq01 p75 scale-target + rotation-spike rescue | 17.591 / 25.462 / 62.545 m | 10.9364% | **0.02025 deg/m** | 216 PnP / 43 Kabsch fallback | Reduces median-target lag in the accelerating highway window while leaving seq00 unchanged |
| seq01 p75 target + `0.80x` moderate-collapse gate | 17.023 / 24.445 / 59.335 m | 10.4991% | **0.02025 deg/m** | 216 PnP / 43 Kabsch fallback | Replaced: rescues weak moderate collapses that missed the older 0.65x gate; seq00 unchanged |
| seq01 + `10 deg` translation-direction rescue | **12.679 / 16.852 / 38.804 m** | **4.9844%** | **0.02025 deg/m** | 216 PnP / 43 Kabsch fallback | Adopt default: clamps weak fast lateral translation outliers while preserving translation magnitude; seq00 unchanged |
| seq01 `--temporal-max-row-delta 5` | Incomplete | Incomplete | Incomplete | n/a | Reject: too strict, timed out at 420 s after reducing early PnP inliers |
| seq01 `--relative-pose-mode kabsch` | Incomplete | Incomplete | Incomplete | n/a | Reject: all-pair Kabsch-first timed out at 420 s |

The adopted motion-scale rescue is deliberately narrow: it requires at least
20 prior motion samples, a recent median translation of at least 1.5 m, weak
consensus (`source != PnP` or PnP inlier ratio below 0.45), and a translation
magnitude outside the recent-median band (`< 0.80x` or `> 1.6x`). It keeps the
direction and rotation from the estimator and rescales only the translation
magnitude. Once triggered, the target is the recent p75 translation rather than
the median, which reduces lag in accelerating highway windows without changing
the seq00 guardrail. This makes it a stopgap for fast forward-motion scale
instability, not a replacement for stronger learned matching or temporal
motion modeling.
The automatic stereo refinement uses the same fast-motion history threshold
and a looser weak-consensus gate (`source != PnP` or PnP inlier ratio below
0.65), then refines only translation against current-frame stereo reprojection
residuals; it preserves the seq00 260-frame guardrail exactly. Rotation-spike
rescue uses the same fast-motion guardrail, requires weak consensus, and clamps
only large relative-rotation angles (`> 1 deg` and `> 3x` the recent median)
back to the recent median. On seq01 this reduces mean/max per-pair rotation
error from `0.2868 / 3.6849 deg` to `0.2077 / 1.1741 deg`. Adding the p75
scale target and widening the weak moderate-collapse gate to `0.80x` then
reduces seq01 public-length translation from `12.3334%` to `10.4991%` while
preserving the `0.02025 deg/m` rotation result.

Translation-direction rescue targets the remaining seq01 failure mode:
weak-consensus fast pairs whose translation magnitude is plausible but whose
lateral component bends the highway trajectory away from the recent direction.
It uses the same 20-sample / 1.5 m fast-motion guard and the same 0.45 weak
PnP-consensus gate, keeps the estimated magnitude, and replaces only the
translation direction when it deviates by more than `10 deg` from the recent
average direction. A `12 deg` threshold was safer than `8 deg` on max-window
error, but `10 deg` kept that worst-window result while improving public
translation to `4.9844%` and ATE to `12.679 / 16.852 / 38.804 m`.

## KITTI Deep Stereo VO Match Quality Trials

Same local KITTI 00 seq00 260-frame diagnostic, using the adopted
`deep_max_features=1500`, `deep_min_confidence=0.15`, `3.32 px` PnP gate, and
1000 relative-pose RANSAC iterations unless otherwise noted.

| Trial | ATE mean / RMSE / max | KITTI `t_rel` | KITTI `r_rel` | max `t_rel` | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline mutual-softmax temperature `25.0` | **2.509 / 2.610 / 3.777 m** | 0.9856% | **0.01419 deg/m** | 2.8988% | Adopted before guarded PnP refinement |
| Temporal row gate `25 px` | Incomplete | Incomplete | Incomplete | Incomplete | Reject: pathological runtime, stopped before 50 pairs |
| Temperature `30.0` | 2.359 / 2.496 / 3.996 m | 1.1677% | 0.01620 deg/m | **2.6578%** | Reject: KITTI mean translation/rotation regress, one Kabsch fallback |
| Temperature `24.0` | Incomplete | Incomplete | Incomplete | Incomplete | Reject: pathological runtime after 100 pairs |
| Temperature `23.0` | 2.318 / 2.425 / 3.353 m | 1.5634% | 0.01792 deg/m | 2.8927% | Reject: ATE improves but KITTI windows regress hard |
| Temperature `22.5` | 2.526 / 2.644 / 4.097 m | **0.9056%** | 0.01713 deg/m | 2.7584% | Reject for balanced default: better translation, worse rotation |
| Temperature `22.5`, PnP gate `3.5 px` | 2.482 / 2.593 / 3.994 m | 0.9095% | 0.01738 deg/m | 2.7274% | Reject: looser gate keeps translation gain but worsens rotation |
| Descriptor clip `0.15` | 2.676 / 2.822 / 4.539 m | 1.1815% | 0.01733 deg/m | 2.9760% | Reject: tighter HOG clipping regresses ATE and KITTI windows |
| Descriptor clip `0.25` | Incomplete | Incomplete | Incomplete | Incomplete | Reject: pathological runtime after 100 pairs |

The `--deep-temperature`, `--deep-descriptor-clip`, and
`--temporal-max-row-delta` knobs remain useful for experiments, but the balanced
default stays at guarded PnP refinement, temperature `25.0`, descriptor clip
`0.2`, high-consensus PnP early-stop, lazy Kabsch fallback, and no temporal row
gate.
