# Frozen weak-angle landmark intervention

Status: predeclared experiment, not a production policy. No intervention run yet.

## Question and limits

The [geometry diagnosis](openloris_observation_geometry_diagnostic.md) found
39 initial landmarks with maximum acute pair angle below 0.1 degrees. They
carry 71.86% of adaptive summed point motion, but only 6.28% of its net cost
reduction. This does not establish their effect on camera trajectory.

Test one intervention: hold those 39 XYZ values at their initial values under
the previously failing adaptive column-scaled LM policy. Compare against the
same adaptive policy without fixed points, and retain legacy matrix-free as
the quality threshold. Using only legacy as the intervention arm would poorly
test the observed large-motion regime: its corresponding motion sum was only
0.032 m. Fixing complete XYZ also constrains transverse motion and treats
uncertain initial triangulation as exact; it may bias cameras and worsen quality.
This is a falsifiable constraint experiment, not a literature claim that weak
points should generally be fixed.

## Frozen inputs and configuration

- Reuse PR #87's original 1k input, rig calibration and combined source SHA;
  do not use a post-optimization model as initialization.
- Select membership from that initial model alone using the already fixed
  maximum acute pair-angle criterion. Independently verify the 39 sorted IDs,
  bind the list to source hashes, and commit it before any intervention run.
- Retain all 4,716 landmarks and 130,900 observations, including observations
  of fixed landmarks. Preserve every image/keypoint/track identity and order.
- Adaptive column-scaled LM: 20 iterations, initial lambda 1e-4, PCG maximum
  512, relative tolerance 1e-8, absolute tolerance 1e-12, restart zero.
  Preserve PR #87's adaptive update rule, scaling clamps, feasibility checks
  and all other defaults. No robust kernel, calibration change or pose prior.
- Fixed rig frame zero remains the anchor; only the 39 landmark constraints
  differ from the adaptive control. No thresholds or solver settings may be
  selected using GT or changed after seeing the intervention result.

## Implementation boundary

Use the existing fixed-landmark API and adaptive optimizer. The existing
rig-factor synthetic regression already exercises fixed XYZ with adaptive
LM. Add only an opt-in, bounded, source-hash-bound ID-list input to the
comparison driver, with duplicate/unknown/malformed/hash mismatch rejection.
Log actual fixed count and list hash. Preserve the option-absent path and
verify its models and numerical traces against the recorded controls.

Do not add pair searches to the optimizer, global pair state, a new library
API or an extra full model clone. The ID list is small; constraints remove
117 scalar unknowns, but do not remove observation traversal. No speed or
memory improvement is assumed. Verify fixed XYZ before and after optimization
and after serialization; preserve source coordinates if formatting would
otherwise alter fixed values.

The frozen list is
[m8-openloris-weak-angle-fixed-landmarks-v1.txt](../benchmarks/electro/m8-openloris-weak-angle-fixed-landmarks-v1.txt),
SHA-256 `a5b12954dd0a2f6c21d41ad5150c5012cc797059b193be95181cae2d8773feb2`.
Its format is `SOURCE_SHA256 <combined hash>`, then `COUNT <N>`, then strictly
ascending `LANDMARK <u64>` rows. Blank lines and full-line comments are allowed;
the file is capped at 64 KiB and the declared count at 8,192. Hash mismatch,
unknown/duplicate IDs, malformed rows and count mismatch must fail before
creating solver output. This opt-in is restricted to adaptive column-scaled
matrix-free mode; fixture export and other solver combinations reject it.

After binary certification, the intervention adds only the list option to
the adaptive control command (paths below are placeholders, not a run record):

```bash
target/release/examples/compare_rig_bundle_adjustment \
  --model /path/to/frozen-1k/model \
  --rig-manifest /path/to/frozen-1k/rig-manifest.txt \
  --solver matrix-free --matrix-free-column-scaling \
  --matrix-free-adaptive-damping \
  --pcg-max-iterations 512 --pcg-relative-tolerance 1e-8 \
  --fixed-landmark-ids benchmarks/electro/m8-openloris-weak-angle-fixed-landmarks-v1.txt \
  --out-dir /path/to/new-intervention/model
```

## Runs and decision gates

After review and scoped tests, certify one binary. Run serially: unchanged
legacy control, unchanged adaptive control, adaptive with the 39 fixed points,
and an exact repeat of that intervention. Controls must match prior models
and numerical traces; the intervention repeat must match its model and trace.
Record wall time and peak RSS for each process with the existing timing scope.

Before post-only GT scoring, require complete identity/support, calibration,
anchor and fixed-XYZ preservation, one supported component, all positive-depth
observations, finite nonincreasing full squared cost and an accepted update.
Use the existing 308-image scoring set and unchanged scorer/calibration files.
Require RMSE <=0.026608055174816774 m, p95 <=0.04109998478546261 m and mean
observation error <=0.691588658326868 px (the legacy thresholds). Lower cost
than adaptive is not required or expected from a constrained optimum; any
claimed quality improvement must be established by the independent scores.

Failure stops this candidate before atlas/10k; do not relax thresholds or
expand the fixed population. Success only permits a separately predeclared
larger-scale test, not default or README promotion. Native mapper/E2E speed,
10k accuracy/RSS, scale tiers, restart and 100k I/O remain full-goal gates.
