# Unordered Structure-from-Motion Benchmark

The [EuRoC SfM benchmark](euroc_sfm_benchmark.md) reconstructs from an **ordered**
video: temporal frame→frame matches give forward tracks, and rectified stereo
gives metric scale for free. That is the easy half of structure-from-motion. The
hard half — the one COLMAP is actually known for — is reconstructing from an
**unordered** image set: no temporal order, no known overlap graph, and (with a
single camera) no metric scale. This benchmark measures that path.

## What "unordered" requires

A photo collection hands you `N` images and nothing else. The pipeline has to:

1. **Discover the view graph.** Which images even overlap? A VLAD vocabulary
   over all SuperPoint descriptors gives each image a global descriptor; the
   top-K most similar images per image become candidate pairs
   (`visloc_vision::place_recognition`). This is `O(N·K)`, not the `O(N²)`
   exhaustive match.
2. **Verify each pair geometrically.** Cross-checked brute-force + Lowe-ratio
   descriptor matching, then an essential-matrix RANSAC keeps only the inliers
   (`visloc_vision::two_view`).
3. **Grow one reconstruction incrementally.** Seed from a well-conditioned pair,
   then repeatedly register the next image by PnP, triangulate the tracks two
   registered views now share, and bundle-adjust — the COLMAP mapper loop
   (`visloc_slam::incremental_sfm`, `pipelines/slam/src/incremental_sfm.rs`).

The core is ~600 lines of pure Rust orchestrating the parts visloc-rs already
had (two-view geometry, PnP RANSAC, VLAD retrieval, Schur-complement BA); the
[`unordered_sfm_demo`](../examples/unordered_sfm_demo.rs) example wires VLAD
retrieval + verification on top of file-backed deep features and exports a
COLMAP model.

### The seed has to have parallax, not just overlap

The first non-obvious lesson: the **strongest-overlap pair is the wrong seed.**
On an orbit the most-matched pair is two *adjacent* low-parallax frames, whose
tiny baseline makes triangulation depth-unstable — seed there and the
reconstruction has no well-conditioned points to register the third image
against, and it stalls at two cameras. So seed selection walks pairs in
descending match count but accepts one only when enough of its correspondences
actually triangulate to well-conditioned points (the same parallax + cheirality
+ reprojection gate the rest of the pipeline uses). This is the incremental-SfM
analogue of COLMAP's essential-vs-homography seed test.

## Result (EuRoC V2_03, unordered)

The cleanest validation reuses the **V2_03 Vicon-Room orbit** — the same capture
whose *ordered* stereo SfM produced the crisp 3D Gaussian Splat in the
[EuRoC SfM benchmark](euroc_sfm_benchmark.md). We take its **left camera only**,
strided to **31 images over frames 0–150**, shuffle away all order, and hand them
to the unordered pipeline as a bare photo set.

| Metric | Value |
| --- | ---: |
| Input images (left camera, strided) | 31 |
| View-graph candidate pairs (VLAD top-10) | 207 |
| Geometrically verified pairs | 207 |
| **Images registered** | **31 / 31** |
| Reconstructed multi-view tracks | 616 |
| Observations | 8,513 |
| **Mean reprojection error** | **0.58 px** |

Sub-pixel reprojection from a monocular, orderless input is already a strong
internal-consistency signal. But the decisive check is **external**: align the
30 recovered camera centres to the trusted *ordered* stereo reconstruction (the
crisp-3DGS model) with a similarity (Umeyama Sim(3)) transform.

| Sim(3) vs ordered reconstruction | Value |
| --- | ---: |
| **Camera-centre RMSE** | **1.08 cm** (median 0.53 cm, max 3.44 cm) |
| Trajectory extent | 0.65 m |

**The unordered, monocular, orderless reconstruction reproduces the ordered
stereo reconstruction to ~1 cm** — about 1.7 % of the trajectory extent. The
pipeline rediscovered the entire overlap graph from VLAD alone and recovered the
same camera geometry (up to the expected global scale, `s ≈ 0.11`, since a single
camera has no metric reference). That is the COLMAP-grade result the SfM pillar
was missing for photo collections. The reconstruction is deterministic —
track ids are assigned in a stable order, so the figures above reproduce
run-to-run.

## A real photo collection, and the PnP conditioning fix it surfaced

EuRoC frames-treated-as-unordered are a fair test of the *orderless* machinery,
but a genuine photo set with mild lens distortion and no temporal relationship
is a stronger one. We ran the pipeline on the **COLMAP South Building** example
dataset (128 JPEGs, 3072×2304, `SIMPLE_RADIAL`), undistorting the SuperPoint
keypoints to a pinhole with the dataset's own intrinsics, and compared the
recovered poses to COLMAP's *own* sparse reconstruction (its `sparse/images.txt`)
by Sim(3) on the camera centres.

High-resolution images need a looser pixel gate — `--max-reproj` scales the
reprojection threshold with image size (4 px on a 752-px EuRoC frame is ~16 px on
a 3072-px photo).

This dataset first **surfaced and fixed a real bug in the DLT PnP solver**: it did
not Hartley-normalise the 3D points, so when points sit far from the origin
relative to the scene scale (a 2-view SfM seed has depth ≫ baseline) the linear
system was badly conditioned and returned a garbage pose even from an all-inlier
sample; and it resolved the projection's overall sign by determinant only, which
could place every point *behind* the camera (reprojection ∞). Both are fixed,
with a regression test, in `crates/vision/src/pnp` — and the fix raised the EuRoC
registration to 30 / 31 as a side effect.

### P3P: registering the planar façade the DLT could not

Even with the conditioning fix, the DLT stalled at **2 / 128** images on South
Building: the 6-point DLT is a *linear* solver and is **degenerate on coplanar
points**, and a building is mostly flat façade. So the third image, registered by
PnP against a near-planar seed, never got a usable pose and the reconstruction
never grew. This is the textbook reason COLMAP and every serious SfM frontend use
a *minimal geometric* solver — P3P — inside the registration RANSAC.

visloc-rs now ships **Grunert's Perspective-Three-Point solver**
(`crates/vision/src/pnp/p3p.rs`, the default `PnpSolver::P3p`): from three world
points and their image bearings it forms a quartic in one length ratio, solves it
by the companion-matrix eigenvalues, recovers the three camera-frame depths for
each real root, and reads off the pose by absolute orientation (Kabsch) — well-posed
for any three non-collinear points whether or not the scene is planar. Swapping it
in for the minimal solver:

P3P **unlocks the planar façade**: the reconstruction grows across the whole
collection instead of stalling at the seed. The same regression tests cover the
general-3D, coplanar, and far-from-origin-seed cases.

### Closing the precision gap: scale-gauge fixing + iterative track filtering

P3P got South Building *registered* (126 / 128) but, at first, only roughly: the
monocular model sat ~1 m (median ~29 cm) from COLMAP with a ~22 px mean
reprojection. Two coupled problems caused that, and both are now fixed.

1. **The monocular bundle adjustment did not pin scale.** A monocular
   reconstruction has *seven* gauge freedoms — six for the rigid SE(3) frame plus
   one for global scale — but the BA fixed only a single pose, leaving the scale
   direction of the normal equations singular. One solve from a perturbed state
   tolerates that, but re-optimising from a converged state lets scale drift and
   the reconstruction **collapse**. The fix anchors scale by also fixing the
   registered pose with the longest baseline to the anchor (`run_bundle_adjustment`),
   exactly as COLMAP fixes its first two cameras.
2. **Union-find tracks were contaminated.** A loop-inconsistent match chain can
   merge two distinct 3D points into one track; its BA'd point then fits none of
   its observations. A **post-BA filtering pass** (`track_filter_iterations`, the
   default) strips every observation that reprojects worse than the gate after the
   global BA and re-optimises, iterating a few rounds. Crucially it never un-poses
   an image, so the **registered-image count is invariant** — it only cleans
   structure, and on a clean reconstruction it is a near-no-op.

Together they turn the rough registration into a COLMAP-grade reconstruction:

| South Building (128 JPEGs, monocular) | DLT | P3P | **P3P + gauge-fix + track filter** |
| --- | ---: | ---: | ---: |
| Images registered | 2 / 128 | 126 / 128 | **128 / 128** |
| Multi-view tracks | — | 21,316 | 22,024 |
| Mean reprojection | — | 22.8 px | **2.0 px** |
| Sim(3) camera-centre RMSE vs COLMAP | — | 106 cm | **0.59 cm** (median 0.47 cm, max 1.12 cm) |

**The orderless, monocular reconstruction now reproduces COLMAP's own model of a
real 128-photo building to 0.59 cm — 0.1 % of the 11 m trajectory extent**, with a
completely independent frontend (SuperPoint, not COLMAP's SIFT). The same two
improvements register all 31 / 31 EuRoC V2_03 images at 0.58 px reprojection and a
1.08 cm Sim(3) RMSE (median 0.53 cm). That is COLMAP-grade unordered SfM on a
genuine photo collection — the precision gap the previous revision flagged as
future work is closed.

## Reproduce

```sh
scripts/run_unordered_sfm_benchmark.sh \
    --feat-dir /path/to/V2_03/left-features \
    --ordered-colmap /path/to/ordered/v203_sfm_colmap
```

The runner symlinks frames 0–150 (stride 5) of the left-camera SuperPoint
features into an unordered set, runs `unordered_sfm_demo` to build the COLMAP
model, and (when `--ordered-colmap` is given) runs `scripts/compare_sfm_sim3.py`
to report the Sim(3) camera-centre RMSE against the ordered reconstruction.

The left-camera features are the same `frame_NNNNNN_left_features.txt` files the
[loop-closure](euroc_loop_closure_benchmark.md) and ordered-SfM benchmarks
already export with the repo's SuperPoint helper; point `--feat-dir` at that
directory. Intrinsics default to the V2_03 rectified pinhole
(`752×480`, `fx=fy=436.24`, `cx=364.44`, `cy=256.95`); override with the
`--width/--height/--fx/--fy/--cx/--cy` flags for another camera.

The same `unordered_sfm_demo` runs on any photo set — point `--features-dir` at a
directory of per-image deep-feature files, supply the camera intrinsics, and the
COLMAP model it writes feeds directly into a 3DGS / MVS pipeline.
