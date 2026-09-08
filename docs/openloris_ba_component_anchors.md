# Native BA component anchoring — unmeasured candidate

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
