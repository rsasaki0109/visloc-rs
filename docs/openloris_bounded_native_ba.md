# Bounded native BA policy — unmeasured candidate

## First 1k measurement

Release `b011cde` produced exact champion model bytes for the same-binary
Legacy control and both policy repeats. Each policy run selected direct on
all 86 calls and QR on zero calls. Mapper seconds: Legacy 4.417592, policy
4.277948 / 4.513326. Process peak RSS KiB: 81,644 / 81,928 / 81,748.
This shows small-window nonregression, not a speed or memory improvement.
Champion quality metrics are inherited by byte identity, not newly scored.
The >64 QR branch is entirely untested by these native runs; larger-tier
quality, end-to-end and memory gates remain open.
[Commands and exact-byte audit](../benchmarks/electro/m8-openloris-bounded-native-1k-v1.json).

## 2.5k paired check on existing sparse7n input

The existing M5 sparse7n snapshot was held identical across Legacy and two
policy runs; it differs from the 1k ANN input, so this is only a within-tier
BA comparison. Models are byte-identical, but only 1,224/2,500 images and
612/1,250 frames register. Each policy run selects direct on all 103 calls;
zero calls exercise large QR. Mapper times are 5.879777 s Legacy versus
6.028910 / 6.042272 s policy, peak RSS 158,472 / 158,628 / 158,612 KiB.
This is nonregression of an insufficient-coverage result, not a successful
tier gate, speedup, COLMAP comparison or large-QR quality test. Diagnose the
input graph / unregistered coverage before claiming larger-tier progress.
[Exact commands and audit](../benchmarks/electro/m8-openloris-bounded-native-2500-sparse7n-v1.json).

Read-only coverage diagnosis: the 16,321 nonempty accepted image-pair edges
project to one connected component containing all 1,250 rig frames. Registered
frames are exactly 0–611, with 37 verified pairs crossing to the unregistered
region. Mapper diagnostics report 603 zero-support, 22 below-PnP-support and
13 eligible-but-unregistered frames (619 lack required sensor support, an
overlapping classification). Pair connectivity is not metric-track or PnP
support. Investigate boundary 3D/track support rather than assuming a disconnected
input graph. [Audit and limitations](../benchmarks/electro/m8-openloris-2500-sparse7n-connectivity-v1.json).

Boundary audit: the 37 crossing pairs contain 998 accepted pair matches, all
same-camera temporal edges. Matching their registered endpoints to exported
3D associations gives frame 612 eight feature/point associations per sensor,
and frame 613 ten per sensor. These are final-model associations before
conflict handling and geometry verification, not the in-loop PnP cache or
guaranteed inliers. Therefore neither zero raw connectivity nor missing a
camera alone explains this boundary. Inspect actual registration rejection
at 612/613 before changing thresholds or enabling recovery.
[Pair/3D association audit](../benchmarks/electro/m8-openloris-2500-boundary-support-v1.json).

Actual registration debug (same certified binary, model-byte-exact) reports
frame 612 with 25 correspondences / two sensors and frame 613 with 29 / two
sensors; both return `pnp=estimation-failed`, not the mapper's sensor or inlier
gate. The generalized estimator already includes per-sensor P3P hypotheses
in addition to DLT. Next distinguish hypothesis failure from insufficient
pooled inliers internally, without relaxing gates or adding duplicate P3P.
[In-loop boundary evidence](../benchmarks/electro/m8-openloris-2500-boundary-pnp-v1.json).

`VISLOC_SFM_DEBUG_PNP_HYPOTHESES` now enables one scalar summary per eligible
generalized-PnP call: successful DLT hypothesis count, successful central
sensor reports (not the internal number of P3P roots), best pooled inliers,
minimum sample support and prior presence. It retains no correspondence or
hypothesis history, does not change RNG draws/scoring/refinement, and requires
native byte-parity verification before interpreting measured output. Six
generalized-related tests pass. This is diagnostic instrumentation, not a fix.

Measured `015ab1b` is model-byte-exact to Legacy. Frame 612 generates 26 DLT
hypotheses and two central reports; frame 613 generates 11 and two. Both best
pooled scores have only five inliers against six required. Candidate generation
is not absent; investigate 2D/3D consistency or independently verified support
without lowering the acceptance threshold. [Audit](../benchmarks/electro/m8-openloris-2500-pnp-hypotheses-v1.json).

The opt-in diagnostic additionally reports each successful central sensor
estimate's own inlier count and its pooled/own-sensor inlier counts after rig
conversion. This separates weak per-camera fits from disagreement across
sensors without changing sampling or choosing a different hypothesis.
Native validation of these additional fields remains pending; the recorded
`015ab1b` result above predates them.

The objective remains native quality, speed and bounded 10k memory, not forcing
every small BA window through an iterative solver. Same-state evidence shows
117 unavailable QR steps on the Legacy path; paired small-window measurements
also favor Legacy. An explicit size policy is therefore the next candidate.

`--ba-backend bounded-direct64-qr` selects before the first solve: at most 64
variable poses use the existing caller-configured direct backend, larger
systems use compact QR. The selection is logged. There is no failure-triggered
retry on another backend, no default change and no tolerance/LM change. Both
arms validate the shared fixed-calibration pure-visual eligibility contract.
Fixed-state checks remain enabled. The 64 boundary is a fixed resource cap,
not a GT-tuned accuracy threshold; it bounds a six-DoF pose matrix to 384×384
(1,179,648 bytes for one dense f64 matrix, not total process memory). Pose-point
storage can still scale with observations/landmarks, so a measured RSS gate
remains mandatory. No global all-pose pair graph is authorized by this policy.

[Ceres upstream solver documentation](https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/docs/source/nnls_solving.rst)
distinguishes factorization and iterative solvers and describes quadratic
Schur storage as limiting direct dense methods to smaller camera problems.
This supports size-aware selection in general, not our exact threshold or
claims of improved quality. The source was checked 2026-09-08; the rendered
documentation timed out, so its upstream source was inspected instead.

Required validation:

1. Exact boundary tests (64 direct, 65 QR), fixed state and invalid-input
   rollback, strict CLI parsing; no selector should depend on solver failure.
2. Same-binary 1k Legacy control and policy repeats: model parity, backend
   selection coverage, independent quality gates and timing. Passing only
   the direct arm does not validate QR at larger sizes.
3. Existing 2.5k/5k/10k native tier gates with selection coverage and 2 GiB cap.
   The large QR arm is still unproven; reject on quality or RSS failure.
4. Full native end-to-end and repeat/restart/100k I/O requirements remain;
   a saved-feature replay cannot close them. README stays unchanged until
   measured COLMAP comparison supports a claim.

This candidate does not establish faster-than-Legacy small-window BA or solve
the previously rejected large-scale quality result. PR #98's shadow diagnostic
is a separate pending change; this branch is stacked on it.
