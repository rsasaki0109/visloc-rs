# Localizing a photo whose focal length is unknown

A query photo without EXIF (a screenshot, a messaging-app copy, a crop) has
no trustworthy focal length. Localizing it against a map with a guessed focal
length does not just lose inliers: a P3P/PnP RANSAC fits a pose to the
distorted rays anyway, and on real data that pose can pass an inlier-count
gate while being far off.

`LocalizationPipeline::localize_with_unknown_focal` estimates the focal length
from the 2D-3D correspondences themselves with P4Pf, then runs the usual pose
estimator and quality gate with it:

```rust
let out = pipeline.localize_with_unknown_focal(
    &query,                      // query.camera: size and principal point; its focal is ignored
    &map,
    &descriptor_store,
    &P4PfRansacConfig::default(),
);
let focal_px = out.camera.map(|c| c.intrinsics().unwrap().0);
let result = out.result;         // gated exactly like localize_with_descriptor_store
```

It assumes square pixels, no lens distortion and a known principal point (the
image centre is the usual choice). The four-point minimal solver
(`visloc_vision::pnp::p4pf_minimal`) is vendored from
[OpenMVG-rs](https://github.com/openMVG/openmvg-rs) (MIT OR Apache-2.0), where it
is checked bit for bit against a Julia reference; `estimate_pose_with_focal`
wraps it in a seeded RANSAC and a Levenberg-Marquardt refit of rotation,
translation and focal length.

## Measured: leave-one-out on the browser SfM's sample

The 38 ETH3D *courtyard* photos the browser SfM ships (800x534, true focal
length 440 px). For each photo, the map is rebuilt from the other 37 with the
known focal length; the held-out photo's SIFT descriptors are matched to the
map points (ratio 0.8, cross-checked) and localized three ways, 4 px inlier
threshold, at least 30 inliers to count as localized:

- **prior**: P3P RANSAC with the focal length the browser SfM gives a photo
  without EXIF, 1.2 x the long side = 960 px;
- **p4pf**: `estimate_pose_with_focal`, focal length estimated;
- **oracle**: P3P RANSAC with the true 440 px.

The error is the distance from the camera centre in the full 38-photo model,
after a similarity alignment of the two models on their common photos, as a
percentage of the scene's mean camera radius.

| | localized | median centre error | |
|---|---|---|---|
| prior (960 px) | 34 / 38 | **351 %** | every "localized" pose is wrong |
| **p4pf** | **36 / 38** | **1.03 %** | median focal error 0.29 % |
| oracle (440 px) | 36 / 38 | 1.13 % | |

P4Pf without any focal length matches the oracle that knows it. The guessed
focal length is the dangerous case: 34 photos pass the inlier gate with poses
1.3-4.6 scene radii off. The two photos no method localizes have 43 and 57
correspondences.

Reproduce (from `bindings/sfm-web`):

```bash
cargo run --release --example unknown_focal_loo -- web/samples/courtyard --focal 440
```

<details><summary>Per-photo output</summary>

```text
38 photos, full model 38 registered; true focal 440 px, no-EXIF prior 960 px
photo        corrs | prior: inl  err | p4pf: inl  focal   err | oracle: inl  err   (err: % of scene radius)
DSC_0286.jpg   680 |  576 160.55 |  626   446.1   1.65 |  626   1.13
DSC_0287.jpg   831 |  641 161.38 |  784   441.9   0.33 |  785   0.72
DSC_0288.jpg  1245 |  958 155.07 | 1198   440.7   0.09 | 1198   0.21
DSC_0289.jpg  1185 |  925 155.59 | 1139   441.9   0.40 | 1140   0.73
DSC_0290.jpg  1083 |  870 153.96 | 1054   438.5   0.78 | 1054   0.29
DSC_0291.jpg  1145 |  840 153.55 | 1099   440.5   0.19 | 1100   0.52
DSC_0292.jpg  1143 |  814 148.03 | 1090   442.2   0.28 | 1091   0.58
DSC_0293.jpg   972 |  780 150.47 |  929   438.8   0.89 |  931   0.53
DSC_0294.jpg   795 |  644 150.02 |  740   439.7   0.29 |  740   0.22
DSC_0295.jpg   752 |  587 153.26 |  692   438.8   0.63 |  692   0.51
DSC_0296.jpg   195 |   64 126.06 |  150   440.2   0.45 |  151   2.10
DSC_0297.jpg   103 |   49 192.32 |   70   439.4   0.31 |   70   0.17
DSC_0298.jpg    72 |   30 161.75 |   35   429.6   3.01 |   35   0.88
DSC_0299.jpg   104 |    -      - |   59   472.0  10.41 |   61   2.60
DSC_0300.jpg   169 |   52 350.96 |  131   436.9   2.26 |  132   1.64
DSC_0301.jpg   172 |   68 378.86 |  151   439.0   2.00 |  151   1.53
DSC_0302.jpg   346 |   87 458.52 |  289   438.4   2.74 |  290   1.94
DSC_0303.jpg   331 |   73 443.70 |  275   440.6   1.54 |  275   1.33
DSC_0304.jpg   301 |   74 426.90 |  249   440.4   1.03 |  250   2.33
DSC_0305.jpg   231 |   59 438.82 |  173   439.3   1.03 |  173   0.56
DSC_0306.jpg   100 |    -      - |   38   443.3   4.09 |   38   3.98
DSC_0307.jpg    57 |    -      - |    -        -      - |    -      -
DSC_0308.jpg    43 |    -      - |    -        -      - |    -      -
DSC_0309.jpg   186 |   67 163.47 |  143   441.9   1.06 |  143   0.90
DSC_0310.jpg   519 |  373 162.16 |  478   442.2   0.70 |  478   0.56
DSC_0311.jpg   272 |   84 346.17 |  211   437.8   2.38 |  211   1.69
DSC_0312.jpg   562 |  208 366.15 |  510   440.6   0.83 |  510   1.27
DSC_0313.jpg   482 |  158 364.30 |  433   441.2   1.06 |  434   0.79
DSC_0314.jpg   273 |   68 365.04 |  229   442.0   0.52 |  229   0.58
DSC_0315.jpg   314 |   98 420.24 |  263   441.2   0.71 |  264   1.88
DSC_0316.jpg   173 |   42 410.25 |  115   438.9   1.08 |  115   0.64
DSC_0317.jpg   438 |  315 442.15 |  390   441.3   0.73 |  391   4.05
DSC_0318.jpg   406 |  143 422.70 |  350   438.6   1.61 |  351   0.47
DSC_0319.jpg   342 |   63 223.07 |  265   439.3   1.06 |  266   1.39
DSC_0320.jpg   272 |  102 399.15 |  211   439.4   2.19 |  212   1.52
DSC_0321.jpg   232 |   84 424.68 |  168   440.1   0.83 |  169   4.63
DSC_0322.jpg   200 |  101 403.47 |  151   441.4   1.06 |  151   1.76
DSC_0323.jpg   355 |  169 464.96 |  312   443.6   3.01 |  313   1.57
prior : localized 34, median centre error 350.96% of scene radius
p4pf  : localized 36, median centre error 1.03% of scene radius
oracle: localized 36, median centre error 1.13% of scene radius
p4pf focal: median error 0.29% of the true focal length
```

</details>

## Where it does not help: SfM

In the incremental mapper, a wrong focal-length prior is already corrected by
bundle adjustment with focal-length refinement: on the same 38 photos, priors
from 120 px to 3000 px all converge to 440.6-440.9 px with 38/38 registered
(only an 8000 px prior fails, 11/38). So the mapper's registration is left
unchanged; P4Pf is for a single query that has no bundle adjustment to fix
its intrinsics.
