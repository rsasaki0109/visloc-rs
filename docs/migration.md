# Migration Guide

`visloc-rs` is still pre-1.0. This guide records the intended migration path toward a stable 1.0 API so applications can adopt the library without depending on accidental module layout details.

## Recommended Import Surface

Application code should prefer:

```rust
use visloc_rs::prelude::*;
```

The prelude contains the common map, query, pose, localization, tracking, mapping, SLAM, fusion, and small IO entry points. Explicit module paths remain supported for narrower imports:

```rust
use visloc_rs::io::colmap::ColmapMapProvider;
use visloc_rs::vision::matching::BruteForceMatcher;
```

Root-level item re-exports such as `visloc_rs::Camera` and `visloc_rs::Pose`
remain available for convenience, but new docs and examples should treat the
prelude and the crate modules (`visloc_rs::core`, `visloc_rs::localization`,
`visloc_rs::vision`, and so on) as the canonical import paths.

For v1.0, the prelude should remain additive where practical. New common entry points may be added, but existing prelude items should not be removed without a documented migration.

## Feature Support

The stable lightweight surface is `--no-default-features`, default features, and
`image-io`. These are the Tier 1 configurations documented in
[feature_matrix.md](feature_matrix.md) and checked by `scripts/check_feature_matrix.sh`.
ONNX Runtime and CUDA paths are opt-in deployment features; they should not be
used to justify visual-only benchmark claims unless the benchmark registry
records the feature flags, model hashes, command, and hardware.

## Map-Based Localization

The stable direction is to build localization around these boundaries:

- `VisualMap` stores reusable cameras, keyframes, landmarks, and observations.
- `QueryImage` or `Frame` stores query-side keypoints and descriptors.
- `LandmarkDescriptorStore` stores external descriptors for COLMAP-style maps.
- `localize`, `localize_with_descriptor_store`, and `LocalizationPipeline` run map-based localization.
- `LocalizationResult` carries success, pose, inliers, reprojection errors, and diagnostics.

Prefer `LocalizationPipeline` when an application needs custom candidate selection, matching, or robust pose estimation. Prefer `localize` only for simple in-memory maps where descriptors are embedded in `Landmark`.

## COLMAP Maps

Use `ColmapMapProvider` for reusable SfM maps:

```rust
let provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
    "path/to/colmap_text_model",
    "path/to/landmark_descriptors.txt",
)?;
```

COLMAP sparse models do not contain feature descriptors. Applications should load descriptors from a separate store or attach them to landmarks before localization. See [colmap_compatibility.md](colmap_compatibility.md) for supported model formats and current limitations.

## Tracking And Priors

For image sequences, prefer `Tracker` or `ImageTracker` over manually carrying the last pose between independent `localize` calls. Tracking APIs expose state transitions, relocalization, quality gates, stats, and pose-prior diagnostics.

External GNSS, odometry, or VIO hints should enter through `LocalizationPrior`, `FramePriorSource`, `MeasurementBuffer`, and `LocalizationPriorProvider`. This keeps visual-only localization independent from future tightly-coupled fusion backends.

## Experimental Layers

The following layers are useful but still expected to evolve before 1.0:

- `LocalMappingPipeline`
- `OnlineSlamPipeline`
- covisibility local BA and pose-graph refinement configuration/results
- `FramePriorSource`
- covariance and timestamp helper types

Use them for experiments and demos, but avoid treating their exact field layout as frozen until the 1.0 API review is complete.

## Breaking-Change Policy Before 1.0

Before 1.0, breaking changes should be rare and should satisfy at least one of these conditions:

- They fix incorrect coordinate-frame semantics.
- They remove a misleading API that implies full SLAM behavior.
- They make a core trait usable by a real example or test.
- They simplify a public API before it becomes stable.

When a breaking change is made, update this guide with the old API, the replacement, and the reason for the change.

## Known Migration Notes

### Prefer The Root Prelude

Older examples used imports such as:

```rust
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, QueryImage, VisualMap};
use visloc_rs::localize;
```

New application examples should use:

```rust
use visloc_rs::prelude::*;
```

The explicit module paths still work. The prelude is now the recommended entry point for common application code.

### Keep SLAM Expectations Explicit

`OnlineSlamPipeline` is currently a foundation-layer composition over tracking
and local mapping. It exposes loop-closure candidates, optional covisibility
local BA, optional pose-graph refinement, VI initialization / local VI-BA
building blocks, and relocalization hooks, but these are not a production full
SLAM guarantee. Applications that only need map-based localization should keep
using `LocalizationPipeline` and `Tracker`, and applications that enable the
SLAM-stage options should keep their benchmark claims tied to the exact
configuration and registry artifacts that were run.
