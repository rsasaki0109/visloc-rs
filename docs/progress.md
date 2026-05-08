# Progress

This file tracks the project milestone completion used in development updates.

## Current Development Completion

**Deep VO / Loop Close completion: 45%**

This score means the core extension boundaries exist, but the project has not
yet reached a real learned-frontend sequence demo or full loop-closure backend.

Completed pieces:

- `VisualOdometryFrontend` and `VisualOdometryPriorProvider` boundaries exist.
- VO-derived pose priors can narrow tracking candidates.
- Externally generated two-view match files can be parsed.
- Online SLAM composition exists over tracking and local mapping.
- Loop-closure candidates can be detected from shared verified landmarks.
- Loop-candidate HTML/SVG reporting exists for synthetic sequence demos.

Remaining pieces before this milestone is considered complete:

- Feed a real classical or learned two-view frontend into the tracking demo.
- Convert external two-view matches into a measured VO estimate, not just a
  file-backed import boundary.
- Run a public sequence demo that shows denser correspondences and smoother
  frame-to-frame motion.
- Add stronger geometric verification and diagnostics for loop candidates.
- Add pose-graph constraint hooks while keeping global optimization optional.
- Keep full pose-graph optimization out of the core until the candidate layer is
  proven by demos and tests.

## Reporting Rule

Development updates should report this value as:

```text
Deep VO / Loop Close completion: 45%
```

Increase the number only when a runnable example, test, or documented API
materially advances the Deep VO or loop-closure path.

## Rubric

- **0-20%:** map-based localization only; no sequence, VO, or loop direction.
- **20-40%:** tracking, local mapping, and SLAM composition boundaries exist.
- **40-60%:** VO frontend boundaries, external match IO, loop-candidate
  detection, and visible loop reports exist.
- **60-80%:** real external classical or learned frontend drives sequence
  tracking, and public demos show correspondences, pose continuity, and loop
  candidates clearly.
- **80-100%:** loop constraints, pose-graph hooks, regression datasets, and
  stable interfaces exist, while heavy runtimes remain optional integrations.
