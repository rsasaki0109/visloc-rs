# Native BA component anchoring — mixed quality, not promoted

## First paired 5k result

Same-binary control matches historical Legacy bytes. Both candidate repeats
match exactly and register all 5,000 images, adding 39 component anchors each.
RMSE improves 0.477216 → 0.261284 m and maximum 10.094671 → 1.773457 m.
However p95 worsens 0.478983 → 0.490866 m and raw mean reprojection worsens
0.831377 → 0.837364 px. This does not pass an all-quality nonregression gate.
Every image has support; depth and fixed rig calibration are valid. Published
track components change from 2495/2/2/1 frames to 2499/1 (singleton 1616).
Mapper seconds: control 53.417580, candidates 53.641363 / 62.541108.
No speedup, large-tier or native-E2E claim. Keep default off; do not select
alternative anchors or thresholds using GT. Diagnose remaining geometric
weakness before extending or promoting this arm.
[Commands, repeat audit and independent scores](../benchmarks/electro/m8-openloris-5000-component-anchors-v1.json).

The certified 5k diagnostic found 306/20,000 pose/solve records without any
path to a fixed pose through actual BA observations. The five largest
anchor-distance changes all occurred in these components. This motivates
an explicit `--ba-anchor-disconnected-components` arm, default off.

Before solving, build a landmark-star graph from accepted observations.
For each component without an existing fixed pose, fix its lowest frame ID
at its current pose. Existing fixed poses are preserved. Pose IDs without
observations are singleton components. Selection is deterministic and does
not use GT, trajectory jumps, reprojection threshold sweeps or solve failure.
Graph storage is linear in observations, landmarks and poses; no pose clique
or dense global matrix is added. Existing solver policy remains unchanged.

This removes a component's rigid gauge but is not a general rank or quality
guarantee: monocular scale, weak geometry and incorrect initial placement
can remain. Ceres documents SfM gauge ambiguity and the distinction between
structural and numerical rank deficiency in its
[official covariance documentation](https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/docs/source/nnls_covariance.rst).
That source motivates handling gauge explicitly; it does not certify our
frame-selection rule or this candidate's reconstruction quality.

Validation: test independent/stereo-only/no-observation components,
existing anchors and input-order invariance; then native fixed-state checks.
Run a same-binary Legacy control and two candidate repeats with unchanged
5k inputs. Audit registration, depth, calibration, reprojection, whole-model
GT scoring and repeatability. Do not remove outlier images or separately
align disconnected track components. If useful, run the independent tier
ladder and 10k; 5k is a derived mapper-only slice, not native E2E evidence.
Frame 1416 already deforms while connected, so this is not presumed sufficient.
