# Multi-view stereo (`visloc-mvs`)

`visloc-mvs` turns a COLMAP sparse model with undistorted (pinhole) images
into a dense point cloud: a depth and normal map per image by PatchMatch,
fused into `fused.ply`. It runs on the CPU, or on the GPU through wgpu with
`--features gpu`.

```text
cargo run --release -p visloc-mvs --features gpu --example mvs -- \
    --sparse <dir with cameras.txt, images.txt, points3D.txt> \
    --images <image dir> --out <dir> --gpu \
    [--max-image-size 3200] [--sources 10] [--iterations 3] \
    [--geom-passes 2] [--geom-iterations 3] [--scales 3] [--no-planar-prior] \
    [--view-selection confidence|sampled] [--mesh]
```

The image names in `images.txt` are relative to `--images`. The cameras must
be `PINHOLE` or `SIMPLE_PINHOLE`: undistort the images first.

## Method

Depth estimation follows ACMH and ACMM (Q. Xu and W. Tao, "Multi-Scale
Geometric Consistency Guided Multi-View Stereo", CVPR 2019), implemented
from the paper:

- **Hypotheses.** Each pixel holds a plane, a unit normal facing the camera
  and its offset, so a hypothesis spreads to a neighbour as a plane, not as
  a depth.
- **Propagation.** Red-black checkerboard updates. Each pixel takes the
  lowest-cost plane of eight regions around it: four V shapes of 7 pixels
  and four strips of 11 pixels along the axes, all of the other colour.
- **View selection.** The eight planes' costs in every source view vote. A
  view counts when more than `n1 = 2` of them are below
  `0.8 exp(-t² / 90)` (`t` the iteration) and fewer than `n2 = 3` above
  1.2. Its weight is the mean of `exp(-m² / (2 · 0.3²))` over its good
  costs. The previous iteration's most important view keeps a double
  weight. `--view-selection sampled` follows ACMM's released code
  instead: those weights, times a prior from the four adjacent pixels'
  selected views, are a distribution that 15 draws sample, and a view's
  weight is its count. Views without weight are not matched.
- **Refinement.** A random and a perturbed depth and normal, combined with
  the current ones into six more planes.
- **Matching cost.** One minus a bilateral-weighted NCC over every other row
  and column of an 11×11 window, warped by the plane's homography. The
  window skips its centre, so where every sample's grey level is far from
  the centre's, all the weights vanish and the pixel does not match.
- **Planar prior** (ACMP: Q. Xu and W. Tao, "Planar Prior Assisted
  PatchMatch Multi-View Stereo", AAAI 2020; at every scale by default,
  `--no-planar-prior` leaves it out). After the photometric pass, each
  view's credible pixels (the lowest-cost pixel of each 5×5 block, if its
  cost is below 0.1) are triangulated in the image (Delaunay). The three
  back-projected corners of each triangle span the plane that is the
  prior of the pixels inside it. A planar prior pass then starts from the
  photometric map, or from the prior where the cost is not credible, and
  ranks a pixel's hypotheses by
  `exp(-c²/0.18) · (0.5 + exp(-Δd²/(2σd²)) · exp(-Δθ²/(2σθ²)))`: `c` the
  cost, `Δd` and `Δθ` the depth and normal angle to the prior, `σd` 1/64 of
  the depth range, `σθ` 5°. Where weak texture leaves the costs unable to
  tell planes apart, the hypothesis closest to the prior wins. Where a
  pixel has a prior, the refinement draws its random depth within 3σd of
  the prior's and its random normal around the prior's, as ACMP's code
  does.
- **Geometric consistency** (`--geom-passes`, 2 by default). Further
  passes start from the previous depth maps and add 0.2 times each source
  view's forward-backward reprojection error, truncated at 3 pixels.
- **Multi-scale** (three scales by default, `--scales 1` runs one), in the
  order of ACMMP's released code (Q. Xu, W. Kong, W. Tao and M. Pollefeys,
  "Multi-Scale Geometric Consistency Guided and Planar Prior Assisted
  Multi-View Stereo", TPAMI 2022): at every scale of the image pyramid a
  photometric pass, the planar prior pass and the geometric passes. A
  finer scale's photometric pass starts from the coarser maps, upsampled
  (joint bilateral: each pixel takes the plane of the best-weighted coarse
  neighbour). Its detail restorer lets a pixel leave its upsampled plane
  only for one whose cost is below that of the plane at infinity (the
  rotation's homography) by 0.1. Where weak texture leaves every plane
  about as costly, the coarser estimate stays; elsewhere the pass
  re-estimates at the finer scale. That is what ACMM's and ACMMP's code
  compute: their baseline plane is the upsampled depth with no normal.
  ACMM's paper compares with the upsampled plane's own cost instead
  (`--restorer start`), which holds planes that are slightly off: on a
  synthetic plane a start 2% too deep stays 2% too deep, where the plane
  at infinity lets it converge to within 0.05%. Our first multi-scale
  scheme, an independent photometric pass at each finer scale kept where
  it beat the upsampled plane by 0.1, lost accuracy on detailed scenes
  for that reason (below).
- **Source views and depth ranges.** Up to 10 source views per image by
  MVSNet's score over the shared sparse points (a Gaussian in the
  triangulation angle around 5°). The depth range comes from the image's
  sparse points: COLMAP's 1% and 99% quantiles, widened by 25%.
- **Fusion.** A pixel's 3D point is kept when a source view agrees within
  2 pixels of reprojection error, 1% of relative depth and 10° of normal
  angle; the agreeing pixels are averaged into one point and not used
  again. ACMM's and ACMP's code ask for two agreeing views: with these
  depth maps one view scores higher (below), and `--min-consistent 2`
  restores two. As in ACMMP's code, the agreeing views must also agree
  well on average: the mean of `exp(-(e + 200 d + 10 a))` over them must
  reach 0.15, with `e` the reprojection error in pixels, `d` the relative
  depth difference and `a` the normal angle in radians
  (`--min-mean-consistency`). ACMMP uses 0.3. On the 13 scenes at 3200 px
  0.15 adds 0.23 point of F1 and 0.3 loses 0.38 (Fusion settings).
  ACMMP's code also keeps the reference pixel's point instead of the
  average (`--no-average`); that changes F1 by 0.1 point or less, so it
  stays off. The checks
  against the source maps run in parallel first; the raster-order pass
  that builds the points re-checks only the sources that agreed, so the
  cloud is a sequential fusion's (byte-identical on pipes at 1600 px, in
  2.7 s instead of 5.1 s on four threads).

The GPU path runs the same per-pixel update as compute shaders, with the
same sampling regions, view selection and random numbers. The source images
are half-float texture layers that the texture unit samples bilinearly, as
ACMM samples CUDA textures. The GPU differs from the CPU path only by
rounding: on ETH3D pipes at 400 px its F1 score is 34.44% against the
CPU's 34.47%.

**Memory.** A view keeps its colours, 3 bytes a pixel; its grey levels
are the colours' luma, computed when the view is matched. A depth map
keeps a depth and a normal, 16 bytes a pixel, and its costs only with
`--save-maps` (no pass or fusion reads them). A geometric pass releases
the previous pass's maps as it goes: a view's normals once it is
re-estimated, its depths after the last view that uses it as a source.
Fusion drops each map after the last view that reads it. So one set of
maps is held at a time, not two, and the output does not change: on
pipes, `fused.ply` is byte-identical on the CPU and the GPU, with one
scale and with three. With the images pre-resized to 800 px (one
iteration per pass, CPU), peak memory falls from 305 to 218 MiB with one
scale and from 364 to 229 MiB with three. At that size glibc keeps some
freed memory: 175 MiB with one scale under
`MALLOC_MMAP_THRESHOLD_=131072`.

On ETH3D facade (76 images at 3200 px, an NVIDIA L4) peak memory falls
from 24.2 to 12.0 GiB with one scale (in 10:16 instead of 11:40) and from
28.7 to 12.4 GiB with three. The F1 scores at 2 cm, 64.96% and 62.03%,
are within 0.03 points of the earlier runs' (64.96% and 62.06%). The
shaders did not change, but that pod had another NVIDIA driver (580.178
instead of 595.91), and the clouds differ slightly: 77.75 million fused
points instead of 77.66 million with one scale.

## Mesh

`--mesh` also writes `mesh.ply`. Each view's depths that a source view
agrees with (fusion's check, whether or not the agreeing pixel went into
a point) are fused into a truncated signed distance field on a sparse
grid, after the last view that reads the view, and the zero level set is
extracted with surface nets (the TSDF of `gsplat_photos`' mesh, now in
`visloc-gsplat-core`). The voxel is one pixel's footprint at the median
over the views of their sparse points' median depth (`--mesh-voxel`,
`--mesh-voxel-pixels`). Pixels at least twice as far go to a second
level four times coarser (`--mesh-levels`, 2 by default). Pixels without
such a depth are unknown rather than empty, a cell whose corners are not all
known still gets a vertex from its known edges (`--mesh-full-cells`
requires all eight), and components under 50 triangles are dropped
(`--mesh-min-component`).

On pipes at 1600 px (the saved depth maps of an earlier run, the earlier
voxel rule below and one level; F1 at 2 cm,
accuracy / completeness):

| Output | F1 | Accuracy | Completeness |
|---|---|---|---|
| `fused.ply` | 74.8% | 95.6% | 61.5% |
| `mesh.ply`, defaults | 68.1% | 94.7% | 53.2% |
| all eight corners known (2-pixel voxel, 2 views per voxel) | 36.7% | 99.3% | 22.5% |
| components of 10 triangles kept | 70.6% | 93.0% | 56.9% |

Requiring every corner loses most of the surface: a pixel whose agreeing
source pixel already went into a point fails fusion's check, so the
depth maps have holes a pixel wide, and one unknown corner drops a cell.
Thin structures (pipes' pipes) and far surfaces, whose depth noise
exceeds the voxel, stay fragmented. A coarser level for far pixels did
not help on pipes, whose surfaces are mostly within 2 m of a camera. On
pipes the mesh adds 1.5 s of fusion and 6.6 s of extraction, and 1.0 GiB
of peak memory (5.3 million triangles, 118 MB).

The voxel used to be the footprint at the median of the views'
depth-range midpoints. Those ranges reach the far 99% quantile plus 25%.
On meadow the midpoint is 40.5 m against a median depth of 7.9 m, so
the voxel came out five times too coarse (4.6 cm at 3200 px; F1 42.2%
at 2 cm in r40, against the cloud's 76.2%). Meadow's 800 px maps,
meshed with two-pixel voxels (F1 at 2 cm; the cloud scores 67.5%):

| Voxel | F1 | Accuracy | Completeness | Blocks | Peak memory |
|---|---|---|---|---|---|
| depth-range midpoint (18 cm) | 3.8% | 59.2% | 2.0% | 11,353 | 487 MiB |
| median depth (3.6 cm), one level | 55.8% | 64.2% | 49.4% | 58,413 | 929 MiB |
| median depth, two levels (the default) | 58.1% | 73.7% | 47.9% | 27,096 | 548 MiB |
| median depth, three levels | 58.1% | 73.7% | 47.9% | 25,342 | 567 MiB |

On office's 800 px maps the second level changes F1 by less than 0.01
point (51.8%) and saves 4% of the memory.

On all 13 scenes at 3200 px (r45, L4, commit `023e741`,
`--mesh-voxel-pixels 2`) the mesh scores F1 79.3% at 2 cm (accuracy
88.5%, completeness 72.5%) against the cloud's 83.5%. On the six scenes
r40 meshed with the old voxel it rises from 72.6% to 78.8%:

- meadow 42.2% → 68.9%;
- terrace 83.2% → 90.1%;
- pipes 79.2% → 81.0%;
- office 66.2% → 68.2%;
- kicker 80.1% → 80.6%;
- relief 84.5% → 83.7%.

The finer voxel costs memory: peak 6.1 GiB on pipes against 3.8 GiB,
15.3 GiB on facade.

## ETH3D high-resolution benchmark

Scores come from the official ETH3D multi-view evaluator
(`ETH3DMultiViewEvaluation`) against the laser scans of the training
scenes, at the benchmark's 2 cm tolerance, with the scenes' ground-truth
camera poses (`dslr_calibration_undistorted`).

### All 13 training scenes on a GPU

One run on an NVIDIA L4 pod at commit `2c9cebe` with what are now the
example's defaults: images at 3200 px, 10 source views, three iterations
per pass, three scales (800, 1600 and 3200 px) in ACMMP's order, each
with a photometric pass, the planar prior pass and two geometric passes.
The same run estimated every scene with one scale too (`--scales 1`, the
previous default). F1, accuracy and completeness are for three scales and
the default fusion (one agreeing source view within 10°); the other F1
columns are for one scale (the same fusion) and for three scales fused
within 15°. Times cover the whole `mvs` run: loading, depth maps, a
fusion and writing the cloud and the maps. (Facade ran on a second pod,
the same GPU type and driver, after its upload failed.)

| Scene | Images | F1 | Accuracy | Completeness | F1, one scale | F1, 15° | Time | Time, one scale |
|---|---|---|---|---|---|---|---|---|
| pipes | 14 | 84.4% | 93.9% | 76.7% | 81.2% | 84.1% | 1:53 | 1:40 |
| meadow | 15 | 76.0% | 84.9% | 68.8% | 63.9% | 76.2% | 2:34 | 2:27 |
| terrace | 23 | 91.0% | 94.8% | 87.5% | 90.0% | 90.7% | 3:43 | 3:12 |
| office | 26 | 79.0% | 76.6% | 81.7% | 78.6% | 77.8% | 3:47 | 3:19 |
| kicker | 31 | 82.3% | 78.9% | 85.9% | 77.6% | 81.3% | 4:54 | 4:24 |
| relief | 31 | 87.8% | 91.6% | 84.3% | 87.9% | 87.4% | 5:20 | 4:31 |
| relief 2 | 31 | 86.5% | 89.3% | 83.9% | 86.9% | 86.1% | 5:24 | 4:26 |
| courtyard | 38 | 90.4% | 90.0% | 90.7% | 89.1% | 90.4% | 6:05 | 5:07 |
| playground | 38 | 69.8% | 77.2% | 63.7% | 67.4% | 69.5% | 6:28 | 5:45 |
| terrains | 42 | 91.2% | 90.9% | 91.5% | 91.3% | 91.0% | 6:54 | 6:09 |
| delivery area | 44 | 90.1% | 89.0% | 91.1% | 89.6% | 89.8% | 6:44 | 5:40 |
| electro | 45 | 89.4% | 94.5% | 84.9% | 88.7% | 89.2% | 6:55 | 6:03 |
| facade | 76 | 67.5% | 74.4% | 61.7% | 65.4% | 67.3% | 13:46 | 11:53 |

- **Mean over the 13 scenes:** F1 83.5%, accuracy 86.6%, completeness
  81.0%, against 81.4% (87.6%, 76.8%) with one scale. With the fusion's
  mean-consistency check, on since (Fusion settings), the same maps
  score 83.7% (88.7%, 79.5%). Over the same
  scenes at 2 cm, ACMMP reports F1 83.42%, HPM-MVS 84.58%, ACMP 79.79%,
  ACMM 78.86% (accuracy 90.67%, completeness 70.42%) and COLMAP 67.66%
  (91.85%, 55.13%) (arXiv 2303.09758, 2210.07582); later methods reach
  84 to 89% (arXiv 2503.13721).
- **Three scales against one:** +2.1 points on average, most in weak
  texture: meadow +12.1, kicker +4.7, pipes +3.2, playground +2.4 and
  facade +2.1. The detailed scenes stay within half a point (relief
  −0.1, relief 2 −0.4, terrains −0.1), where the earlier multi-scale
  scheme lost 3.6 to 6.3 points (below). Completeness rises by 4.1 points
  for 1.0 point of accuracy.
- **The planar prior's draws** around the prior change little: with one
  scale, 81.36% against 81.44% in the previous run (commit `21b6596`,
  below), within 0.3 point on every scene.
- **Fusion:** 10° beats 15° with three scales too (83.1% at 15°).
- **Time:** 74 minutes for all 13 scenes with three scales, 65 with one,
  from 1:53 (14 images) to 13:46 (76 images). On facade the finest scale
  takes 2:27 for the photometric pass, 2:27 for the planar prior pass and
  2:08 and 2:05 for the geometric passes; the two coarser scales take
  2:26 together. Peak memory 14.6 GiB on facade (13.7 GiB with one scale;
  `--save-maps` keeps the costs).

### End to end, against COLMAP

The whole pipeline from the photos: visloc-rs (`gsplat_photos`' SfM, then
`visloc-mvs`) against COLMAP 4.2.1 built with CUDA (SfM, then its
`patch_match_stereo` and `stereo_fusion`), each on an NVIDIA L4 pod (the
same GPU type), visloc-rs at commit `b05aeca`. Both take the images at
3200 px, one pinhole camera for all of them (the scene's most used
ground-truth camera, intrinsics held fixed), GPU SIFT and exhaustive
matching; each model is then moved onto the ground-truth camera centres
by a Sim(3) (`scripts/align_colmap_model.py --centres`, refitted
without centres beyond three times the median error), and the clouds are
scored like the ground-truth-pose runs. visloc-mvs runs with its
defaults, COLMAP with `image_undistorter --max_image_size 3200`,
`patch_match_stereo --PatchMatchStereo.geom_consistency true` and
`stereo_fusion`'s defaults.
COLMAP's pod stopped at its 7-hour cap after 10 scenes. F1 at 2 cm
(accuracy / completeness), the median camera-centre error after the
alignment (visloc-rs / COLMAP), and the SfM + MVS times:

| Scene | Images | Ground-truth poses | visloc-rs | COLMAP | Centre error | visloc-rs time | COLMAP time |
|---|---|---|---|---|---|---|---|
| pipes | 14 | 84.4% | 89.3% (96.4 / 83.1) | 37.5% (98.2 / 23.1) | 0.11 / 0.11 cm | 0:06 + 2:23 | 0:12 + 17:28 |
| meadow | 15 | 76.0% | 26.0% (61.3 / 16.5) | 31.2% (83.6 / 19.2) | 2.70 / 1.00 cm | 0:07 + 3:06 | 0:13 + 17:38 |
| terrace | 23 | 91.0% | 90.0% (95.6 / 84.9) | 80.5% (94.7 / 70.0) | 0.33 / 0.22 cm | 0:09 + 4:36 | 0:24 + 28:17 |
| office | 26 | 79.0% | 76.5% (73.0 / 80.5) | 37.9% (92.3 / 23.8) | 0.18 / 0.14 cm | 0:07 + 4:33 | 0:19 + 31:33 |
| kicker | 31 | 82.3% | 80.3% (77.7 / 83.1) | 49.9% (95.4 / 33.8) | 0.32 / 0.11 cm | 0:11 + 5:59 | 0:37 + 37:26 |
| relief | 31 | 87.8% | 85.1% (90.9 / 80.0) | 68.7% (95.9 / 53.5) | 0.11 / 0.10 cm | 0:13 + 6:14 | 1:07 + 38:55 |
| relief 2 | 31 | 86.5% | 83.7% (85.8 / 81.6) | 68.0% (93.1 / 53.5) | 0.10 / 0.10 cm | 0:13 + 6:25 | 0:53 + 38:38 |
| courtyard | 38 | 90.4% | 85.0% (90.3 / 80.3) | 75.1% (91.0 / 63.9) | 0.68 / 0.45 cm | 0:16 + 7:00 | 1:17 + 49:04 |
| playground | 38 | 69.8% | 60.4% (63.4 / 57.8) | 51.1% (76.7 / 38.3) | 0.21 / 0.27 cm | 0:16 + 7:49 | 0:42 + 50:21 |
| terrains | 42 | 91.2% | 86.6% (82.0 / 91.7) | 73.9% (94.0 / 60.9) | 0.23 / 0.20 cm | 0:18 + 8:14 | 0:48 + 56:04 |
| delivery area | 44 | 90.1% | 83.9% (78.0 / 90.7) | – | 0.30 cm | 0:16 + 8:22 | – |
| electro | 45 | 89.4% | 80.5% (87.5 / 74.6) | – | 0.32 cm | 0:20 + 8:07 | – |
| facade | 76 | 67.5% | 61.0% (63.8 / 58.4) | – | 0.98 cm | 0:41 + 16:01 | – |

- **Against COLMAP, over the 10 scenes both finished:** F1 76.3% against
  57.4%, higher on 9 of 10 (meadow, below, is the exception), in 58
  minutes against 372 (6.4 times faster). COLMAP's clouds are more
  accurate (91.5% against 81.6%) and far less complete (44.0% against
  73.9%): its fusion keeps a point seen consistently in at least 5
  images. Its MVS takes 17 to 56 minutes a scene; visloc-rs' SfM takes
  6 to 41 seconds and its MVS 2:23 to 16:01.
- **Against the ground-truth poses:** 76.0% over the 13 scenes against
  83.5%. Office, kicker and the reliefs, with one camera, lose 2.0 to 2.8
  points (delivery area 6.2; pipes gains 4.9); the scenes with several
  cameras lose 4.6 to 9.4 (courtyard, playground, terrains, electro,
  facade; terrace 1.0): one camera for every image leaves the
  others' focal lengths and principal points a few pixels off at 3200 px
  (COLMAP's run shares that handicap). The loss is mostly accuracy at the
  tightest tolerance, a misregistration of a few millimetres: on delivery
  area accuracy falls from 77% to 57% at 1 cm but only from 96.7% to
  94.1% at 5 cm, and completeness is unchanged at 5 cm.
- **Meadow** defeats both SfMs: visloc-rs registers all 15 images but
  triangulates 850 points and leaves a 2.7 cm median centre error (one
  image 9 cm off); COLMAP registers 14, one 23 cm off. With 16384
  keypoints instead of 8192 visloc-rs keeps a 25 cm outlier. Its sparse
  points lie in a band along the barrack's wall (none on the grass, roof
  or trees), which leaves the relative rotations about 0.09° off the
  ground truth (pipes: 0.04°). DSC_6541, looking along the wall, comes
  out 1.15° off in both SfMs: its posts and gutter repeat bay after bay,
  and matches one bay apart agree with a slightly wrong pose (82 of its
  268 matches with DSC_6540). The 26.0% above is also one unlucky run.
  The same command on a software Vulkan device instead of the L4 (852
  points against 850) gave a 1.3 cm median centre error, and r46
  (`65ad452`, an L4) fused that model at 3200 px:

  | Poses | Camera-centre Sim(3) | Then ICP onto the scan |
  |---|---|---|
  | Ground truth | 75.2% | 74.4% |
  | visloc-rs SfM (that run) | 58.8% | 70.3% |
  | pycolmap 4.2.1 SfM (CPU SIFT, 14 images) | 68.6% | 71.4% |

  Fusion's reprojection limit does not matter: 3, 4 or 8 px instead of
  2 px change F1 by less than 0.2 point for every pose set. The camera
  centres, almost in one horizontal plane with one 22 cm outlier, fix the
  alignment's tilt poorly. ICP moves the visloc-rs cloud by 0.13° and
  3.1 cm (COLMAP's by 0.07° and 1.2 cm, the ground truth's by 0.02° and
  3 mm). Aligned by the scan, the SfM poses lose about 4 to 5 points to
  the ground truth on meadow.
- **Tools for a next run:** `scripts/eth3d_one_camera.py` resamples a
  scene's images onto its most used camera (`K K_i^-1`, exact for pinhole
  images: patches at the moved observations match with NCC 0.997), and
  `scripts/eth3d_icp_align.py` refines the alignment by ICP against the
  scan; on pipes the latter moves a ground-truth-pose cloud by 1.8 mm
  (F1 74.8% → 74.4% at 1600 px), so it measures little beyond noise there.

### End to end again: one camera per scene, and ICP

r47 (L4, commit `9f2b115`) repeated r39a's run on all 13 scenes with two
changes. Scenes with several cameras were first resampled onto their
most used camera (`scripts/eth3d_one_camera.py`). Every cloud was scored
both as aligned by the camera centres and after ICP onto the scan
(`scripts/eth3d_icp_align.py`), the way Tanks and Temples refines its
alignment. The table gives F1 at 2 cm with visloc-mvs's defaults; r39a's
column is its own run.

| Scene | Cameras (images resampled) | Ground-truth poses | r39a | r47 | r47, ICP | ICP's move | Centre error median / max |
|---|---|---|---|---|---|---|---|
| pipes | 1 | 84.1% | 89.3% | 88.5% | 88.6% | 0.06°, 1.3 mm | 0.11 / 0.4 cm |
| meadow | 1 | 75.2% | 26.0% | 25.5% | 47.2% | 1.09°, 98 mm | 2.70 / 9.5 cm |
| terrace | 4 (15) | 91.2% | 90.0% | 91.2% | 90.4% | 0.07°, 5.3 mm | 0.19 / 0.5 cm |
| office | 1 | 80.2% | 76.5% | 78.1% | 78.4% | 0.17°, 6.3 mm | 0.18 / 0.8 cm |
| kicker | 1 | 83.8% | 80.3% | 81.8% | 82.0% | 0.10°, 6.0 mm | 0.32 / 132 cm |
| relief | 1 | 88.2% | 85.1% | 85.4% | 86.0% | 0.02°, 1.4 mm | 0.11 / 0.3 cm |
| relief 2 | 1 | 87.0% | 83.7% | 84.3% | – | – | 0.10 / 0.2 cm |
| courtyard | 4 (15) | 90.2% | 85.0% | 90.1% | 89.3% | 0.03°, 4.6 mm | 0.40 / 1.7 cm |
| playground | 5 (27) | 69.5% | 60.4% | 61.2% | 62.3% | 0.04°, 1.7 mm | 0.16 / 0.4 cm |
| terrains | 2 (18) | 91.4% | 86.6% | 90.8% | 91.3% | 0.02°, 0.4 mm | 0.11 / 0.2 cm |
| delivery area | 1 | 90.3% | 83.9% | 84.2% | 86.6% | 0.02°, 5.8 mm | 0.30 / 0.5 cm |
| electro | 6 (29) | 89.5% | 80.5% | 75.4% | 77.0% | 0.02°, 2.5 mm | 0.15 / 5.8 cm |
| facade | 5 (27) | 67.3% | 61.0% | 49.0% | 64.2% | 0.04°, 12.2 mm | 1.34 / 4.6 cm |

- **Means:**
  - over the 13 scenes, 75.8% against the ground-truth poses' 83.7%;
  - over the 12 with an ICP score (relief 2's scan holds non-finite
    points, which r47's script did not skip), 75.1% and 78.6% after ICP.
- **The SfM is deterministic on one GPU.** Fused with r39a's rule (no
  mean-consistency check), the seven one-camera scenes give r39a's F1 to
  the tenth.
- **One camera per scene helps four of the six multi-camera scenes and
  hurts two.** With r39a's rule:
  - helped: courtyard 85.0% → 90.2%, terrains 86.6% → 90.5%, terrace
    90.0% → 91.1%, playground 60.4% → 60.7%;
  - hurt: electro 80.5% → 75.2% (four images 1.6 to 5.8 cm and 0.2 to
    0.3° off) and facade 61.0% → 49.2% (median centre error 1.34 cm
    against 0.98).
  - The change of images changes the SfM run. Its gain on the camera
    model is lost in the run-to-run spread on these two.
- **ICP moves most clouds by millimetres and a few hundredths of a
  degree.** Where it moves more, the camera-centre Sim(3) is the
  weakness:
  - meadow (above) moves 1.09°. The cameras' orientations put the
    centres' fit 1.05° off (`align_colmap_model.py --rotations`).
  - facade moves 12 mm and gains 15 points.
- **kicker registers its last image 66° and 1.3 m off**, and stays 2
  points under its ground-truth-pose F1. The structure-less registration
  is now checked against the tracks it continues, which refuses that
  image ([eth3d_auc_benchmark.md](eth3d_auc_benchmark.md#checking-structure-less-registrations)).

### End to end with the structure-less check, three alignments (r49)

r49 (L4, commit `c134a36`) repeated r47 with the structure-less
registration check
([eth3d_auc_benchmark.md](eth3d_auc_benchmark.md#checking-structure-less-registrations)).
Each cloud is scored three ways:
- aligned by the camera centres, as before;
- by the cameras' orientations and centres (`align_colmap_model.py
  --rotations`, the cloud moved by the similarity between the two fits),
  which uses only the ground-truth poses;
- after ICP onto the scan.

F1 at 2 cm:

| Scene | Ground-truth poses | r47 | Centres | Orientations | ICP |
|---|---|---|---|---|---|
| pipes | 84.1% | 88.5% | 88.5% | 88.7% | 88.6% |
| meadow | 75.2% | 25.5% | 25.5% | 63.1% | 47.2% |
| terrace | 91.2% | 91.2% | 91.2% | 90.3% | 90.4% |
| office | 80.2% | 78.1% | 78.1% | 78.8% | 78.4% |
| kicker | 83.8% | 81.8% | 81.9% | 82.0% | 82.1% |
| relief | 88.2% | 85.4% | 85.4% | 85.9% | 86.0% |
| relief 2 | 87.0% | 84.3% | 84.3% | 84.7% | 84.7% |
| courtyard | 90.2% | 90.1% | 90.1% | 90.9% | 89.3% |
| playground | 69.5% | 61.2% | 61.2% | 60.7% | 62.3% |
| terrains | 91.4% | 90.8% | 90.8% | 91.0% | 91.3% |
| delivery area | 90.3% | 84.2% | 84.2% | 86.3% | 86.6% |
| electro | 89.5% | 75.4% | 78.0% | 78.3% | 78.8% |
| facade | 67.3% | 49.0% | 49.0% | 49.4% | 64.2% |
| **Mean** | 83.7% | 75.8% | 76.0% | **79.2%** | 79.2% |

- **The check changes two scenes.**
  - electro registers 41 of 45 images instead of 44, its largest
    centre error 1.8 cm instead of 5.8, and gains 2.6 points.
  - kicker loses DSC_6487 (66° off in r47) and gains 0.1. That image
    sees little of the scan.
  - The other eleven scenes give r47's F1 to the tenth.
- **Aligned by the cameras' orientations, the 13 scenes score 79.2%,
  4.5 points under the ground-truth poses.** That is as high as ICP
  without the scan.
  - meadow: the centres' fit is 1.05° off. The orientations fix it and
    the scene goes from 25.5% to 63.1%, above ICP's 47.2%, which stops
    in a local minimum.
  - office and delivery area gain 0.7 and 2.1 points.
- **facade is the exception.** Its cameras drift a few centimetres along
  the 76-image wall, smoothly (one axis of the centre error runs from −3.5
  to +2.4 cm along the sequence). The
  orientations move its cloud by 0.2 mm, while ICP moves it 12 mm and
  gains 15 points. Refining the focal length and principal point in a
  replay of its matches leaves the drift as it is. The far end of the
  wall is bent away from the rest; the alignment now leaves it out
  ([below](#facades-far-end)).
- **The protocol now aligns by the orientations.**
  `align_colmap_model.py` does so by default (`--centres` for the
  centres' fit). The COLMAP comparison above aligned both by the centres;
  COLMAP's clouds were not kept, so it stays as measured.
- **The mapper's final converged BA** (`ba_global_final_converge_min_frames=0`)
  was replayed on r49's L4 matches (r49a, seeds 0 to 2). It lowers
  meadow's relative-rotation error from 0.13° to 0.10° and changes the
  other scenes within noise, so the default stayed. Once electro's
  rescued images were in, `gsplat_photos` took it up
  ([below](#electro-what-the-unregistered-images-cost-r51)).

### facade's far end

Measured locally on r49a's L4 model of facade (76 images; the same model
as r49's), with the ground-truth cameras and the scan:

- **The far end of the wall is bent.** Aligned on the rest, the
  cameras are within 0.6 cm of the ground truth (median). The 14
  cameras of the last 4 m are 2.5 to 6.5 cm off, mostly along the wall,
  and up to 0.4° off in orientation. One camera-centre fit spread that
  error over the whole wall (median 1.4 cm); ICP moved the cloud 12 mm.
- **It is not the mapper.** COLMAP 4.2.1's incremental mapper (pycolmap)
  on the same keypoints and verified matches bends the same end the same
  way (up to 5.9 cm, against 6.5).
- **It is the optimum of these observations.** A global bundle
  adjustment started from the ground-truth poses converges to the same
  bent shape. Its cost is 7% below the ground truth's. The following
  leave the bend as it is:
  - a Cauchy loss;
  - radial or OpenCV distortion refined (k1 comes out at −6e-5);
  - leaving out DSC_0414 and DSC_0415, whose matches with each other
    fit the ground-truth poses worst of the end's links (0.74 px median
    Sampson error, against 0.13 px between DSC_0356 and DSC_0357).
- **The ground truth is not exact there either.** Triangulated with the
  ground-truth poses, the far end's sparse points need a 3.6 cm shift
  along the wall onto the scan. With the model's own poses they need
  0.3 cm, but they spread more: 36% within 2 cm of the scan, against 45%
  for the shifted ground-truth ones. Part of the model's 5 cm from the
  ground truth there is the ground truth's.
- **The alignment now leaves a bent part out.** `align_colmap_model.py`
  fits the scale and translation on the cameras whose orientations agree
  with the rest, then refits once without centres beyond three times the
  median error. On facade that leaves out 15 cameras: 13 of the far end
  and two at the other.
  The sparse points within 2 cm of the scan go from 43.5% to 50.7%
  (median distance 2.42 → 1.96 cm).
  - The other 12 scenes move by at most 1 mm, except meadow, which moves
    1.6 cm. Against the scan, meadow scores the same either way (39.2%
    and 39.1% of sparse points within 2 cm).
  - The script also normalizes the quaternions now. ETH3D writes six
    digits, and the angles taken by arccos near the identity magnified
    that rounding to about 0.1°. The orientation check had dropped half
    the cameras of relief and delivery area for that noise.

r50 (commit `3178986`) scored meadow and facade again under the new fit.
No L4 was free on RunPod, so it took r49a's L4 SfM models (the same as
r49's) and ran the MVS on an A40. The centre-aligned and ICP scores
repeat r49's within 0.6 points, so the other GPU moves little. F1 at
2 cm:

| Scene | Ground-truth poses | Centres (r49 / r50) | r49's orientation fit | New fit (r50) | ICP (r49 / r50) |
|---|---|---|---|---|---|
| meadow | 75.2% | 25.5% / 25.5% | 63.1% | **64.9%** | 47.2% / 47.3% |
| facade | 67.3% | 49.0% / 48.5% | 49.4% | **70.7%** | 64.2% / 64.2% |

- **facade goes from 49.4% to 70.7%** (accuracy 77.0%, completeness
  65.3%). That is above ICP's 64.2%. It is also above MVS from the
  ground-truth poses (67.3%), whose far-end poses sit a few centimetres
  off the scan.
- **meadow gains 1.8 points.**
- **Over the 13 scenes the mean is 81.0%**, 2.7 points under the
  ground-truth poses' 83.7%. The other 11 scenes keep r49's orientation
  scores, as the new fit moves them by at most 1 mm.

### electro: what the unregistered images cost (r51)

electro trails its ground-truth-pose F1 by 11 points end to end (78.3%
against 89.5% in r49), almost all of it completeness: 66.9% against
83.9% at 2 cm. r51 (L4, commit `f10462d`) took the end-to-end images
(resampled onto one camera) and r49a's SfM model (the same as r49's).
It ran visloc-mvs on four sparse models, F1 at 2 cm:

| Sparse model | Images | F1 (accuracy / completeness) |
|---|---|---|
| Ground-truth poses and ETH3D's points | 45 | 89.4% (95.9 / 83.8) |
| The same | 41 | 80.6% (96.1 / 69.4) |
| Ground-truth poses, the SfM's points | 41 | 80.8% (96.4 / 69.5) |
| The SfM's poses, ETH3D's points | 41 | 78.2% (93.5 / 67.1) |

- **The four images the SfM leaves out cost 8.8 points**, all in
  completeness. They are 1.5% of ETH3D's sparse points, but the MVS
  loses far more surface without them.
- **The sparse points cost nothing.** Depth ranges and view selection
  from the SfM's points do as well as from ETH3D's.
- **The SfM's poses cost 2.4 points** (median centre error 0.15 cm,
  largest 1.8 cm).
- **One camera costs nothing:** 89.4% against r45's 89.5% on the
  original images.

Three of the four images now register
([eth3d_auc_benchmark.md](eth3d_auc_benchmark.md#views-panned-from-one-spot)).
r52 (L4, commit `87f25fe`) ran electro end to end with that mapper. The
SfM registered 44 of 45 images, as the replays had (largest centre error
2.3 cm). F1 aligned by the orientations:

| Run | Images | F1 at 2 cm (accuracy / completeness) | F1 at 5 cm (accuracy / completeness) |
|---|---|---|---|
| r49 | 41 | 78.3% (94.6 / 66.9) | 82.9% (97.5 / 72.1) |
| r52 | 44 | 78.7% (94.8 / 67.3) | 86.4% (97.7 / 77.4) |
| Ground-truth poses (r51) | 45 | 89.4% (95.9 / 83.8) | 95.1% (97.8 / 92.6) |

- **At 5 cm the three images add 5.3 points of completeness.**
- **At 2 cm they add 0.4.** Their poses are 0.6 to 1.5 cm and 0.17 to
  0.27° off (DSC_9292, which sees the same ground, 2.3 cm and 0.32°). At a
  few metres' depth that puts their surfaces 1 to 3 cm off, outside the
  2 cm tolerance.
- **DSC_9291 stays out.**
- **Those poses are not converged.** A BA run to convergence, started
  from the mapper's model or from the ground-truth poses, lands on the
  same poses: DSC_9289, DSC_9290 and DSC_9298 at 0.04, 0.02 and 0.03°
  and 0.8, 1.0 and 0.3 cm. The incremental loop's global BAs stop after
  a step or two, and the pipeline's converged final BA only runs from
  1,000 frames. `gsplat_photos` now runs it on every model
  (`ba_global_final_converge_min_frames=0`).
  - Mapper replays (seeds 0 to 2): electro's relative-rotation error
    goes from 0.040° to 0.031° and its largest centre error from 1.7-2.3
    to 1.3-1.8 cm. meadow's goes from 0.13 and 0.15° to 0.10 and 0.11° on
    seeds 0 and 1 (seed 2 stays at 0.10°).
  - The other scenes change by under a millimetre: delivery area's p90
    centre error goes from 0.39 to 0.45 cm, courtyard's and facade's
    largest errors come down 2 mm. The mapper takes a few seconds more.
- **r53 (L4, commit `371c47e`) measured it end to end.** The SfM again
  registered 44 of 45 images, now with the three at 0.02 to 0.04° and 0.3
  to 1.0 cm (largest centre error 1.8 cm, DSC_9292). F1 aligned by the
  orientations:

  | Run | F1 at 2 cm (accuracy / completeness) | F1 at 5 cm |
  |---|---|---|
  | r52 | 78.7% (94.8 / 67.3) | 86.4% |
  | r53 | **84.5%** (94.3 / 76.5) | **92.5%** |
  | Ground-truth poses (r51) | 89.4% (95.9 / 83.8) | 95.1% |

  electro now trails its ground-truth poses by 5.0 points, against 11.1
  in r49. With r53's electro, the 13-scene mean is 81.5% (81.0% before).
  The other scenes keep their r49 and r50 runs: replays put them within a
  millimetre, except meadow, whose rotations improve, but which is not
  re-measured.
- **What is left: DSC_9291 and DSC_9292** (local diagnosis on r49a's
  L4 matches and the converged replays, seeds 0 to 2). The data do not
  pin these two down, and no mapper setting seen so far would:
  - **DSC_9291** has 3,698 keypoints, against 8,000 to 11,000 for its
    neighbours. Its only reliable matches are 172 to DSC_9292, 8 cm away
    (about 1° of parallax at that depth, below the structure-less
    parallax rule), and about 11 to DSC_9293. The 101 matches to DSC_9289,
    DSC_9290 and DSC_9298 are 82 to 98% wrong under the ground truth
    (median Sampson error 14 to 23 px): repeated structure. Its
    structure-less tries see 0 to 13 verified tracks and 17 to 159
    inliers with parallax out of about 200, and are rejected.
  - **DSC_9292** registers on 83 to 85 points at a median depth of 18.6 m,
    seen in four views with a median triangulation angle of 3.5°. With the
    points held fixed, 1 px of noise moves its centre by 0.17 cm, but it
    sits 1.2 to 1.7 cm off. Its points reproject 0.9 px from the
    estimated pose and 3.9 to 4.1 px from the ground-truth pose: the far
    points carry the error, and the pose follows them. Its 172 good
    matches go to DSC_9291, which is not registered, and would add no
    parallax if it were.
  - So the lever is in the matches, not the mapper: more features on
    DSC_9291, or matches between these two and the registered views
    that the current matching does not find (guided matching, for
    example). Neither is tried yet.

### Against ACMMP on the same GPU

ACMMP's released code (GhiXu/ACMMP, built for the L4) on the same
images at 3200 px and ground-truth poses, its input from its
`colmap2mvsnet_acm.py`, against visloc-mvs at commit `183af76` with its
defaults. Each cloud comes from its own pipeline's fusion. F1,
accuracy and completeness are at 2 cm; times cover the whole run
(ACMMP's input conversion included). The pod was lost after six scenes.

| Scene | visloc-mvs F1 (acc. / comp.) | ACMMP F1 (acc. / comp.) | `--geom-iterations 2` | Mesh | Time, visloc-mvs | Time, ACMMP |
|---|---|---|---|---|---|---|
| pipes | 84.4% (93.8 / 76.7) | 82.2% (95.4 / 72.2) | 83.6% | 79.2% | 2.4 min | 7.4 min |
| meadow | 76.2% (85.0 / 69.1) | 72.6% (87.8 / 61.9) | 75.3% | 42.2% | 3.1 min | 9.9 min |
| terrace | 91.1% (94.9 / 87.6) | 90.7% (95.9 / 86.0) | 91.0% | 83.2% | 4.5 min | 16.5 min |
| office | 78.4% (76.3 / 80.7) | 83.7% (87.5 / 80.3) | 78.3% | 66.2% | 4.6 min | 17.4 min |
| kicker | 82.3% (79.1 / 85.8) | 82.7% (85.5 / 80.1) | 81.6% | 80.1% | 5.9 min | 28.2 min |
| relief | 87.8% (91.6 / 84.4) | 87.2% (95.2 / 80.4) | 87.7% | 84.5% | 7.0 min | 30.6 min |
| **Mean / total** | **83.4%** | **83.2%** | 82.9% | 72.6% | **27.5 min** | **110.0 min** |

- **Accuracy and time.** Over these six scenes the two score within 0.2
  point (74.1% against 73.0% at 1 cm, 90.6% against 90.7% at 5 cm), and
  visloc-mvs is 4.0 times faster. ACMMP is the more accurate on every
  scene and visloc-mvs the more complete, by 3.9 points on average.
- **Office.** This is the scene where ACMMP leads, by 5.3 points. The two
  are equally complete, but ACMMP is 11 points more accurate. ACMMP's
  fusion also requires the agreeing views to agree closely on average
  (`--min-mean-consistency 0.3`). On our 3200 px maps that rule takes
  office from 78.4% to 80.1% (see Fusion settings). The remaining 3.6
  points are in the depth maps.
- **`--geom-iterations 2`** (ACMMP's count) saves 13% of the time for
  0.5 point of F1; the default stays 3.
- **Mesh** (`--mesh --mesh-voxel-pixels 2`): 72.6% against the clouds'
  83.4%. Meadow's 42.2% came from its voxel, 4.6 cm. The voxel was then
  taken from the views' depth-range midpoints, which reach far behind
  the scene (terrace got 2.3 cm, office 0.31 cm). It now comes from the
  median depth, with a second, coarser level for far pixels (see Mesh).

### Against APD-MVS on the same GPU (r54)

APD-MVS's released code (whoiszzj/APD-MVS at `38660f3`, built for the
L4's compute capability 8.9) on the same five scenes and ground-truth
poses, its input from its `colmap2mvsnet.py` at half resolution (3104 px)
as its README does for ETH3D, against visloc-mvs's defaults at 3200 px
(commit `74dde86`). Each cloud comes from its own pipeline's fusion. F1 at
2 cm, with accuracy / completeness; the last column is APD-MVS's own
reported F1:

| Scene | visloc-mvs | APD-MVS | APD-MVS reported | Time, visloc-mvs | Time, APD-MVS |
|---|---|---|---|---|---|
| terrace | **91.2%** (95.8 / 87.1) | 88.4% (96.4 / 81.7) | 90.51% | 4:21 | 16:48 |
| office | 80.6% (82.6 / 78.8) | **88.8%** (92.3 / 85.6) | 90.68% | 4:19 | 20:28 |
| kicker | 83.8% (83.4 / 84.3) | **88.4%** (90.0 / 86.9) | 87.38% | 5:29 | 33:47 |
| playground | 69.5% (79.5 / 61.7) | **77.7%** (84.5 / 72.0) | 80.15% | 7:15 | 34:36 |
| facade | 67.3% (76.6 / 59.9) | **72.6%** (80.6 / 66.0) | 75.64% | 15:00 | 1:16:47 |
| **Mean / total** | 78.5% | **83.2%** | 84.87% | 36:24 | 3:02:26 |

- **The gap is real and in weak texture.** APD-MVS leads by 4.6 to 8.2
  points on office, kicker, playground and facade, on accuracy and
  completeness both; visloc-mvs leads on terrace by 2.8 points, on
  completeness. Over the five scenes it is 4.7 points ahead.
- **The reproduction is close to the paper.** 1.7 points below APD-MVS's
  reported mean over these scenes (kicker 1.0 above, the others 1.9 to
  3.1 below); the half resolution and the evaluator's version may account
  for it.
- **visloc-mvs is 5.0 times faster** in total (3.9 to 6.2 times per
  scene). APD-MVS's input conversion adds 0.5 to 2.5 minutes per scene.
- **office loses on accuracy most** (82.6% against 92.3%): the planes
  visloc-mvs puts on weak texture are wrong rather than missing.
  **playground loses on completeness most** (61.7% against 72.0%).

### One scale and the planar prior

One run at commit `21b6596`: one scale, the photometric pass, the planar
prior pass (its random draws not yet around the prior) and two geometric
passes. F1, accuracy and completeness are for the default fusion; the
other columns are the F1 of the run before the planar prior (commit
`3f05992`, the same fusion) and of the same maps fused with 15° and with
two agreeing views at 30°.

- **Mean over the 13 scenes:** F1 81.4%, accuracy 87.7%, completeness
  76.9%.
- **The planar prior raises F1 on all 13 scenes,** from 79.5% to 81.4%
  on average. It adds the most in weak texture, on office (+7.2 points),
  meadow (+3.8) and pipes (+3.4), and 0.1 to 0.3 points on the detailed
  relief scenes and playground. Completeness rises from 73.5% to 76.9% at
  about the same accuracy (88.0%, 87.7%).
- **The angle:** 10° and 15° are close (81.3% at 15°); 10° is higher on
  eight scenes and lower on three, and it is ACMM's and ACMP's angle, so
  it is the default. 30° scores 80.6%. Two agreeing views at 30°
  score 79.3% (accuracy 89.0%, completeness 72.5%).
- **Time:** 77 minutes for all 13 scenes (65 without the planar prior),
  from 1:54 (14 images) to 14:17 (76 images). On facade the photometric
  pass takes 2:56, the planar prior pass 3:19, the geometric passes 2:35
  each and the fusion on the CPU 2:17. Peak memory 13.3 GiB (facade;
  `--save-maps` keeps the costs).

### Without the planar prior

The run before the planar prior (commit `3f05992`, otherwise the same,
fused with 15°) scored F1 79.4% (accuracy 85.9%, completeness 74.8%) in
65 minutes; 77.3% with two views at 30°. Its peak memory was 24.2 GiB
on facade, 12.0 GiB since the maps are released as they go (Memory,
above). Against the first full-resolution run (commit `cf329e2`, before
the NaN fix, the skipped unweighted views and the texture sampling), the
two-view F1 agrees within 0.1 point on each of the 12 scenes that run
covered, which took 53 minutes instead of 99.

### Multi-scale with the paper's detail restorer

The same scenes with our first multi-scale scheme: three scales, two
geometric passes at each and no planar prior, at commit `33b7418`, whose
PatchMatch code is that of `3f05992`. A finer scale ran an independent
photometric pass, kept where its photometric cost beat the upsampled
plane's by more than 0.1. F1, accuracy and completeness are for one
agreeing view within 15°; the last F1 column is the single-scale run
without the prior, with the same fusion.

| Scene | Images | F1 | Accuracy | Completeness | F1, one scale | Time |
|---|---|---|---|---|---|---|
| pipes | 14 | 79.7% | 91.2% | 70.7% | 78.0% | 1:59 |
| meadow | 15 | 68.4% | 81.9% | 58.8% | 61.5% | 2:39 |
| terrace | 23 | 88.7% | 92.7% | 84.9% | 88.7% | 3:50 |
| office | 26 | 65.7% | 64.5% | 67.0% | 70.4% | 4:01 |
| kicker | 31 | 80.0% | 82.3% | 77.8% | 75.1% | 5:29 |
| relief | 31 | 83.9% | 87.8% | 80.3% | 87.5% | 5:21 |
| relief 2 | 31 | 81.9% | 84.7% | 79.2% | 86.7% | 5:24 |
| courtyard | 38 | 86.4% | 87.2% | 85.6% | 88.7% | 6:01 |
| playground | 38 | 66.0% | 73.4% | 59.9% | 67.5% | 6:32 |
| terrains | 42 | 84.2% | 83.8% | 84.5% | 89.4% | 7:19 |
| delivery area | 44 | 81.3% | 79.3% | 83.3% | 87.5% | 7:00 |
| electro | 45 | 86.4% | 93.4% | 80.5% | 86.8% | 7:10 |
| facade | 76 | 62.1% | 71.0% | 55.1% | 65.0% | 13:19 |

- **Mean over the 13 scenes:** F1 78.0%, accuracy 82.6%, completeness
  74.4%, against 79.4%, 85.9% and 74.8% with one scale.
- **It helps weakly textured scenes and costs accuracy on detailed
  ones:** +7.0 points on meadow, +4.9 on kicker and +1.6 on pipes, but
  −6.3 on delivery area, −5.2 on terrains, −4.8 on relief 2 and −4.7 on
  office. The better of the two runs per scene would average 80.5%.
  Upsampled planes that are slightly off cost less than 0.1 more than the
  right ones, so they stayed, and geometric consistency between views
  that all inherited them held them there; ACMMP's order with the
  restorer against the plane at infinity (above) re-estimates them.
- **Fusion:** the variants rank as with one scale: 77.3% with 30°, 76.4%
  without the normal check, 76.0% with two views at 30°.
- **Time:** 76 minutes, 1.2 times the single-scale run. Peak memory
  28.7 GiB (facade), 12.4 GiB since the maps are released as they go.

### Tried and not adopted: HPM-MVS's sampling and KNN prior

Two parts of HPM-MVS (C. Ren et al., ICCV 2023), each added alone to the
default pipeline, on pipes and meadow pre-resized to 800 px (CPU, F1 at
2 cm):

| Run | pipes | meadow |
|---|---|---|
| One scale | 59.0% | 52.3% |
| One scale, NESP sampling | 61.5% | 49.1% |
| Three scales (the default) | 61.5% | 61.0% |
| Three scales, NESP sampling | 62.0% | 49.0% |
| Three scales, KNN prior supplement | 61.5% | 59.3% |

- **NESP** (non-local extensible sampling): candidates from staircases and
  strips starting 5 pixels out, extended outwards where a candidate
  matches badly. It helps pipes and costs meadow up to 12 points of
  completeness-driven F1: no candidate comes from the nearest pixels.
- **KNN prior supplement:** pixels outside the triangulation take a plane
  through three of their six nearest credible pixels. It covers 1 to 2%
  more pixels, mostly with planes that are wrong there.

HPM-MVS pairs them with priors built at three scales in a single-scale
estimation; neither part alone pays here. Those hierarchical priors do
not pay either: on an L4 at 3200 px (commit `8f6a956`), prior passes
with credible-pixel blocks of 20, 10 and 5 pixels at the finest scale
(10 and 5 at the middle one) scored F1 82.4% on pipes, meadow, terrace
and office against 82.6% with one prior pass, within 0.4 point on each
scene, in 1.45 times the time; the run stopped there.

### Tried and not adopted: APD-MVS's deformable patches

`--deformable` adds APD-MVS's adaptive patch deformation (Y. Wang et al.,
CVPR 2023; [`deformable`](../crates/mvs/src/deformable.rs)) at the two
finer scales. Each pass sorts its pixels at its end: a pixel is strong
when its cost along its ray has one clear minimum near its depth. A weak
pixel in the next pass is matched with its own window plus small windows
at strong anchors that RANSAC finds coplanar around it, and it tries
only the anchors' planes and a plane fitted to them. On an L4 at
3200 px, ground-truth poses, F1 at 1, 2 and 5 cm:

| Run | terrace | office |
|---|---|---|
| Defaults (r37) | 84.1 / 91.0 / 95.7% | 68.4 / 79.0 / 88.6% |
| `--deformable`, sorted at each pass's start (`70d438e`) | 76.3 / 83.9 / 90.0% | 64.5 / 75.4 / 85.3% |
| `--deformable`, sorted at each pass's end (`96257a8`) | 82.9 / 90.1 / 95.1% | 65.0 / 76.1 / 86.2% |

- **Sorting.** The first version sorted at the start of each pass. At a
  new scale that meant sorting the coarse depths upsampled. On office at
  400 px the cost curves had their minima 8 to 11 steps off the pixel's
  depth, and 72 to 93% of terrace's pixels came out weak. Sorting at the
  end of each pass, as APD's code does, makes the first pass at a scale
  start from the coarser scale's classes. That brought terrace down to
  30% weak in that pass and recovered 6.2 points of F1.
- **Still below the defaults.** On terrace 40 to 49% of pixels stay weak
  in the later passes. A fifth to a third of the anchored pixels move by
  more than 1% in every pass, against 4 to 13% of the others: the anchor
  planes do not settle. ACMP's planar prior and the detail restorer
  already cover weak texture here. APD's own pipeline has neither; it
  filters strong pixels and refines depths locally after each pass.
  Matching it would mean porting that pipeline, not the patches alone.
- **Time:** about 3.5 times the defaults on these scenes (12 and 14
  minutes with the debug statistics on).

**Two more parts of APD's pipeline** (on top of `--deformable`; the
strong median runs on the GPU too, local refinement on the CPU only;
[notes](apd_mvs_code_notes.md), §5.2 and §5.3):
`--deformable-local-refine` moves each pixel's depth, after every
deformable pass, to the lowest cost within 5 disparity steps along its
ray if that beats its own by 0.1. `--deformable-strong-median` makes the
median filter of a deformable pass leave weak pixels alone and take the
median of a pixel and its strong neighbours (ACMM's 20-pixel pattern)
instead of the 5×5 median of every depth. office pre-resized to 400 px,
ground-truth poses, CPU (4 threads), F1 at 1, 2 and 5 cm:

| Run | F1 | Accuracy / completeness at 2 cm | Time |
|---|---|---|---|
| Defaults | 23.9 / 38.5 / 58.2% | 80.2 / 25.3% | 585 s |
| `--deformable` | 21.9 / 36.3 / 54.5% | 73.9 / 24.0% | 1033 s |
| `--deformable`, local refinement | 21.8 / 36.2 / 54.6% | 74.8 / 23.9% | 1066 s |
| `--deformable`, strong median | 24.0 / **38.7** / 58.3% | 77.6 / 25.8% | 1043 s |
| `--deformable`, both | 23.6 / 38.3 / 58.3% | 77.7 / 25.4% | 1046 s |

- **The strong median recovers what the deformable patches lose** (+2.4
  points at 2 cm), most of it accuracy: the 5×5 median mixed the weak
  pixels' depths into their strong neighbours'. It ends 0.2 point above
  the defaults, within run-to-run noise, in 1.8 times the time.
- **Local refinement does nothing here**, alone or with the median.
- These are 400 px runs on one scene; the strong median now runs on the
  GPU path as well, for 3200 px runs.

### Fusion settings

The same depth maps fused with different rules (CPU runs with one
geometric pass; F1 at 2 cm). One agreeing source view instead of ACMM's
two adds far more completeness than it costs accuracy:

| Run | 30°, two views | any angle, two views | 30°, one view | any angle, one view |
|---|---|---|---|---|
| pipes, 800 px | 53.8% | 56.4% | 67.6% | 68.3% |
| meadow, 800 px | 51.3% | 52.4% | 56.4% | 57.1% |
| pipes, 1600 px | 66.9% | 68.3% | **76.0%** | 75.6% |

On pipes at 1600 px, one view takes completeness from 51.1% to 65.4% and
accuracy from 96.9% to 90.7%. On pipes at 800 px, a stricter depth
(0.5%) or reprojection (1 px) check, or dropping pixels by matching cost,
only lowered F1. At 3200 px (above) a tighter normal check pays: 10° and
15° beat 30°, and on the multi-scale maps 30° beats no check.

ACMMP's mean-consistency rule (`--min-mean-consistency`, with or without
`--no-average`), on the default 3200 px maps of four scenes (r44, L4,
commit `1cbf908`; F1 at 2 cm, accuracy / completeness):

| Scene | Default | 0.3 | 0.3, no average | 0.15 |
|---|---|---|---|---|
| pipes | **84.4%** (93.8 / 76.7) | 83.1% (97.1 / 72.7) | 83.2% | 84.1% (95.2 / 75.3) |
| office | 78.4% (76.3 / 80.7) | 80.1% (88.9 / 72.9) | **80.2%** | **80.2%** (82.7 / 77.8) |
| kicker | 82.3% (79.1 / 85.8) | **84.7%** (88.9 / 80.9) | **84.7%** | 83.8% (83.6 / 84.1) |
| terrace | 91.1% (94.9 / 87.6) | 91.1% (97.4 / 85.5) | 91.1% | **91.2%** (95.8 / 87.1) |
| Mean | 84.0% | 84.8% | 84.8% | 84.8% |

The rule trades completeness for accuracy. It pays on the indoor scenes,
whose wrong depths agree loosely between views (office +1.8, kicker +2.4
points), and costs pipes up to 1.3. On 800 px maps it cost about
10 points.

On all 13 scenes (r45, L4, commit `023e741`, the default maps fused three
ways; F1 at 2 cm):

| Scene | Rule off | 0.15 (the default) | 0.3 |
|---|---|---|---|
| pipes | **84.4%** | 84.1% | 83.1% |
| meadow | **76.2%** | 75.2% | 72.4% |
| terrace | 91.1% | **91.2%** | 91.1% |
| office | 78.4% | **80.2%** | 80.1% |
| kicker | 82.3% | 83.8% | **84.7%** |
| relief | 87.8% | **88.2%** | 87.5% |
| relief 2 | 86.6% | **87.0%** | 86.9% |
| courtyard | **90.4%** | 90.2% | 89.0% |
| playground | **69.8%** | 69.5% | 68.2% |
| terrains | 91.2% | 91.4% | **91.8%** |
| delivery area | 90.0% | **90.3%** | **90.3%** |
| electro | 89.4% | **89.5%** | 88.7% |
| facade | **67.5%** | 67.3% | 66.1% |
| **Mean** | 83.46% (86.6 / 80.9) | **83.69%** (88.7 / 79.5) | 83.08% (91.9 / 76.3) |

At 0.15 the rule is ahead at every tolerance: 73.27% against 73.11% at
1 cm and 91.89% against 91.67% at 5 cm. It gains on eight scenes and
loses at most 1.0 point (meadow) on the others, so it is on by default.
At 0.3 it costs too much completeness on the outdoor scenes.

### Local runs on pipes (CPU)

Before the NaN fix, with two-view fusion:

| Images | Passes | F1 | Accuracy | Completeness |
|---|---|---|---|---|
| 800 px | photometric | 45.8% | 96.7% | 30.0% |
| 800 px | + 1 geometric | 53.7% | 95.3% | 37.4% |
| 1600 px | + 1 geometric | 67.0% | 97.0% | 51% |

Resolution matters most: ACMM and COLMAP are evaluated at up to 3200 px.
