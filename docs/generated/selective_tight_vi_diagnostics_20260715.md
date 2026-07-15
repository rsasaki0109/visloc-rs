# Selective tight-VI diagnostics (2026-07-15)

Status: diagnostic candidate, **not promoted**. The implementation adds
transactional safety and observability, but the tested policy does not pass the
three-sequence accuracy/continuity gate.

## Implemented safety and diagnostics

- Motion-based VI initialization now reports raw rotation, velocity, and
  position IMU residual RMS and can reject promotion on physical-unit bounds
  before the covariance-normalized NIS gate.
- Online tight local VI BA can reject write-back on final IMU NIS per degree of
  freedom. A rejected solve leaves poses, landmarks, velocity, and both bias
  states unchanged.
- Covisibility local BA reports maximum pose correction and supports
  transactional translation/rotation correction gates.
- Covisibility local BA can be restricted to an early-map keyframe count. This
  is an experiment control, not a promoted default.
- The EuRoC demo records each gate in its per-trigger CSV and summary counters.

The implementation retains the full covariance preintegration, covariance
whitening, camera/body extrinsic, velocity, gyro/accelerometer bias, and FEJ
marginalization paths. All new gates default to disabled.

## 300-frame hypothesis screen

The common visual intervention was covisibility BA with at least 30 active
observations. The selective tight-VI policy used raw initializer bounds of
`0.01 rad`, `0.25 m/s`, and `0.08 m`, initializer IMU NIS `<= 20000`, and
joint local-BA final IMU NIS/DoF `<= 5`.

| sequence | arm | tracking | live rigid ATE m | interpretation |
| --- | --- | ---: | ---: | --- |
| MH_01 | control | 0.917 | 0.2603 | visual control |
| MH_01 | covisibility BA | 0.950 | 0.2077 | prefix improvement; tight VI not promoted |
| MH_03 | control | 0.903 | 0.1718 | visual control |
| MH_03 | covisibility BA | 0.923 | 0.1469 | prefix improvement; tight VI not promoted |
| MH_05 | control | 0.920 | 0.3019 | visual control |
| MH_05 | selective tight VI | 1.000 | 0.0590 | 3 local VI writes accepted; later high-NIS solves rejected |

The raw residual gate separates MH_05 (approximately `0.003 rad`, `0.14 m/s`,
`0.02-0.05 m`) from MH_01/MH_03 (approximately `0.04-0.06 rad`,
`0.55-0.78 m/s`, `0.11-0.12 m`). This makes sequence-independent gating
possible without reading a dataset name, but a short prefix is not sufficient
promotion evidence.

## 1,000-frame single-repetition falsification screen

The control values are medians from the existing three-repetition promoted
visual-loop matrix. Candidate rows are single diagnostic repetitions and are
therefore only used to reject unsafe hypotheses, not to claim improvement.

| sequence | arm | tracking | live / final rigid ATE m | d1 RPE m / deg | d10 RPE m / deg | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MH_01 | control | 0.965 | 0.2492 / 0.2543 | 0.1734 / 5.99 | 0.3545 / 14.20 | control |
| MH_01 | covisibility BA, unbounded | 0.958 | 0.2350 / 0.3280 | 0.1834 / 5.39 | 0.5846 / 14.82 | reject |
| MH_01 | covisibility BA, first 30 keyframes | 0.960 | 0.2560 / 0.2932 | 0.1905 / 5.98 | 0.4170 / 14.53 | reject |
| MH_03 | control | 0.872 | 2.5609 / 2.7717 | 0.4272 / 6.29 | 3.2076 / 31.82 | control |
| MH_03 | covisibility BA | 0.877 | 2.4336 / 2.6647 | 0.4260 / 5.77 | 3.2103 / 29.67 | reject: d10 translation and runtime |
| MH_05 | control | 0.889 | 5.1402 / 5.3535 | 0.4153 / 12.46 | 3.3539 / 28.48 | control |
| MH_05 | selective tight VI, NIS/DoF <= 5 | 0.898 | 4.9613 / 5.0925 | 0.4155 / 4.33 | 3.2028 / 25.60 | promising, but d1 translation +0.065% and runtime +82% |

On MH_05, 3 of 100 tight local-BA triggers were accepted and 97 were rejected
by the final NIS gate. The gate prevents repeated unhealthy state write-back,
but does not solve the late visual tracking cliff. On MH_01, limiting visual BA
to the first 30 keyframes still regresses tracking, live/final ATE, and both
translation RPE horizons. Therefore neither visual BA policy is cross-sequence
safe, and no three-repetition promotion matrix was run for a candidate already
falsified by the safety gate.

## Literature interpretation

Primary implementation choices follow Forster et al. for on-manifold IMU
preintegration, VINS-Mono/OKVIS for tightly coupled visual-inertial state and
marginalization, and ORB-SLAM3 for multi-map visual-inertial tracking and
relocalization context. The [ROBOMECH 2026 Docswell overview](https://docswell.com/s/ystk_hara/K4N93D-sfm-vslam-vloc-robomech2026)
is used as a secondary systems checklist: distinguish local BA from global
BA/pose-graph optimization, report ATE together with runtime, retain tracking
interruption evidence, and test learned local features separately. It does not
replace the primary papers or justify promotion by itself.

The next experiment should target the late cliff directly: trigger a bounded
relocalization/map-reseed path from frame-level support and innovation signals,
then compare it against the unchanged promoted visual control. Further tuning
of an always-on local BA is not supported by these results.

## Pose-prior tracking cliff diagnosis and bounded rescue

Frame-level audit identified a concrete non-observation cliff. On MH_05 frame
173 the PnP solution had 306 inliers and a 0.655 inlier ratio, yet tracking was
rejected because its camera centre differed from the fixed motion prior by
`0.311 m`, above the `0.2 m` teleport gate. MH_03 showed the same rejection
class, including a marginal `0.216 m` crossing immediately before natural
recovery.

`TrackingConfig::pose_prior_visual_override` now widens the translation gate
only when all of the following hold: at least 100 inliers, inlier ratio at
least 0.6, mean reprojection error at most 3 px, translation innovation between
1.25 and 2.5 times the ordinary gate, and rotation innovation at most one
degree. The hysteresis excludes MH_03's marginal crossing; the rotation gate
excludes MH_05's second, less-consistent branch. Per-frame CSV and summary
counters record actual activation.

| sequence / horizon | override count | tracking control -> candidate | live / final ATE candidate m | d1 candidate m / deg | d10 candidate m / deg | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MH_01 / 300 | 0 | 0.917 -> 0.917 | 0.2603 / 0.2483 | 0.1629 / 5.30 | 0.2786 / 11.49 | exact metric no-op |
| MH_03 / 300 | 0 | 0.903 -> 0.903 | 0.1718 / 0.2626 | 0.2844 / 5.29 | 0.4197 / 5.46 | exact metric no-op |
| MH_05 / 300 | 1 | 0.920 -> 0.963 | 0.1891 / 0.2105 | 0.1943 / 1.82 | 0.4335 / 3.03 | all accuracy/coverage metrics improve |
| MH_05 / 1000 | 1 | 0.889 -> 0.900 | 5.0743 / 5.2462 | 0.4392 / 4.49 | 3.4381 / 27.48 | reject: translation RPE regresses |

The full-horizon screen proves that accepting the strong visual pose wholesale
is not yet safe: it improves coverage, ATE, and rotation RPE, but changes the
continuation branch enough to worsen translation RPE. No repetition matrix is
claimed for this rejected policy.

An opt-in `motion_vi_raw_residual_activation` gate was also added to
`OnlineSlamCovisibilityLocalBaConfig`. It runs conditioning BA only while
motion-VI initialization is pending and its latest raw residual rejection is
inside configured coarse bounds, then stops on promotion. The MH_05 screen
activated as intended while excluding the measured MH_01/MH_03 residual
profiles, but it either started too late to promote or, with an earlier
initializer trigger, worsened the visual trajectory. It remains diagnostic and
is not part of an adopted configuration. Relaxing the initializer instead was
correctly rejected: the resulting initializer NIS/DoF was approximately
245,816 and all 11 local tight-VI solves were transactionally rejected by the
final NIS gate.

## Full-sequence three-repetition promotion matrix

The bounded rescue was made continuation-safe by estimating a visual
translation covariance from the accepted PnP inlier geometry, fusing it with
the pose prior, and capping the covariance gain at 0.5. The externally reported
pose remains the independently estimated visual PnP pose; only tracker history
and the motion model receive the fused continuation pose. Ill-conditioned
visual Hessians fail closed to the prior.

Continuous covisibility BA was also made support-selective. It activates only
when the earliest keyframe has at most 900 observed landmarks. This is a
sequence-independent structural signal: the measured bootstrap supports are
1273 on MH_01, 1198 on MH_03, and 714 on MH_05. The dense sequences therefore
remain byte-for-byte on the promoted visual control path, while sparse MH_05
receives early visual conditioning before the strictly gated motion-VI
initializer and local tight-VI window.

The final protocol ran the complete EuRoC sequences with one executable/model/
ONNX Runtime hash, three repetitions, and alternating control/candidate order.
The control is the promoted fixed-weight 0.1 SE(3) PnP/PCM/GNC loop-welding
configuration. Medians are shown below; accuracy and continuity values were
identical across all three repetitions.

| sequence | arm | tracking | longest continuity | rigid ATE m | d1 RPE m / deg | d10 RPE m / deg | loop precision | runtime median s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MH_01 | control | 0.905 | 558 | 2.9272 | 0.2538 / 4.7720 | 1.5217 / 21.1040 | 1.000 | 798.3 |
| MH_01 | candidate | 0.905 | 558 | 2.9272 | 0.2538 / 4.7720 | 1.5217 / 21.1040 | 1.000 | 1062.9 |
| MH_03 | control | 0.767 | 215 | 3.6816 | 0.4357 / 6.0837 | 3.2276 / 33.9538 | 1.000 | 604.8 |
| MH_03 | candidate | 0.767 | 215 | 3.6816 | 0.4357 / 6.0837 | 3.2276 / 33.9538 | 1.000 | 705.3 |
| MH_05 | control | 0.843 | 144 | 6.4940 | 0.4730 / 5.0329 | 3.7482 / 27.3704 | 1.000 | 352.0 |
| MH_05 | candidate | 0.884 | 391 | 6.3923 | 0.4506 / 5.0747 | 3.5838 / 27.8397 | 1.000 | 390.5 |

MH_01 and MH_03 candidate trajectories, tracking diagnostics, keyframe
trajectories, final errors, and loop constraints hash exactly to their controls.
On the MH_05 cliff sequence, tracking improves by 4.86%, longest continuity by
171.53%, rigid ATE by 1.57%, and delta-1/delta-10 translation RPE by 4.74% and
4.39%. Delta-1/delta-10 rotation RPE regress by 0.83% and 1.71%; both remain
inside the declared 2% accuracy non-inferiority bound. Runtime is report-only
per the experiment objective, but the dense-sequence motion-initializer attempt
overhead is visible and is not claimed as a speed improvement.

MH_05 initializes at frame 216 in all three repetitions with scale 1.0. Its
initializer state stays inside the configured velocity and bias limits. Of 263
local tight-VI triggers, 3 are accepted and mirrored into the IMU motion model;
260 are transactionally rejected, including 247 by the final NIS gate and 253
by the pose-correction gate (gate reasons may overlap). All three accepted
updates carry successful marginalization priors. This fail-closed state health,
together with the full-sequence safety matrix, promotes the support-selective
candidate. Machine-readable evidence and the exact table are in
[`tracking_cliff_tight_vi_full_3rep_20260715.json`](tracking_cliff_tight_vi_full_3rep_20260715.json)
and [`tracking_cliff_tight_vi_full_3rep_20260715.md`](tracking_cliff_tight_vi_full_3rep_20260715.md).

Reproduce a fresh matrix and summary from PowerShell with:

```powershell
$env:ORT_DYLIB_PATH = 'E:\tools\colmap\bin\onnxruntime.dll'
$env:PATH = 'E:\tools\colmap\bin;E:\tools\venv-cu\Lib\site-packages\torch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;' + $env:PATH
.\scripts\run_tracking_cliff_tight_vi_matrix.ps1 `
  -DatasetRoot 'E:\datasets\euroc_mav\machine_hall' `
  -OutRoot 'E:\visloc_archive\tracking_cliff_tight_vi_full'
python .\scripts\summarize_tracking_cliff_tight_vi_matrix.py `
  --root 'E:\visloc_archive\tracking_cliff_tight_vi_full' `
  --json-out .\docs\generated\tracking_cliff_tight_vi_full.json `
  --markdown-out .\docs\generated\tracking_cliff_tight_vi_full.md
```

The runner refuses incomplete-directory overwrite, alternates variant order by
sequence and repetition, checks executable/model/ONNX Runtime SHA-256 before
every new run, and supports validated `-Resume` after interruption.
