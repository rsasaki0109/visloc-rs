# EuRoC Structure-from-Motion Benchmark

The loop-closure benchmarks ([KITTI](kitti_loop_closure_benchmark.md),
[EuRoC](euroc_loop_closure_benchmark.md)) measure **trajectory** accuracy. This
one measures the **structure** visloc-rs's SfM pillar produces: how close to
COLMAP-grade a single global bundle adjustment can drive the reconstruction, so
the output can feed a downstream 3D Gaussian Splatting / MVS pipeline.

![EuRoC MH_03 SfM reconstruction: merged multi-view track-length histogram (mean 11.3 views per landmark) and mean reprojection error tightening from 4.08 px to 1.04 px after one global bundle adjustment](assets/euroc_mh03_sfm_reconstruction.png)

## Why a per-frame stereo lift is not COLMAP-grade

The streaming stereo VO (and the plain `--colmap-export`) lifts **one fresh
landmark per frame**: every 3D point is observed exactly once. A model like that
has no multi-view constraint for a bundle adjustment to exploit — the points are
independent per-frame depth lifts that overlap into fog, and feeding it to a
3DGS optimizer reconstructs that fog rather than crisp geometry.

A real SfM model shares each physical point across every frame that sees it. The
`--sfm-colmap-out` path builds exactly that:

1. **Merged multi-view tracks.** Chain the temporal (frame→frame) matches into
   forward tracks, so one landmark accumulates the full list of frames that
   observe it (`reconstruct_stereo_vo_with_ba`,
   `pipelines/slam/src/stereo_vo_ba.rs`).
2. **Metric initialisation.** Seed each track's 3D point from its stereo
   observation; the rectified-stereo baseline anchors absolute scale, so no
   extra gauge is needed beyond fixing pose 0.
3. **One global bundle adjustment.** A sparse-Cholesky Schur-complement BA over
   *all* poses and *all* landmarks at once (`pipelines/slam/src/bundle.rs`).
4. **COLMAP export with real tracks.** `write_colmap_reconstruction_for_3dgs`
   writes `cameras.txt` / `images.txt` / `points3D.txt` whose `POINT3D` lines
   carry genuine multi-view `TRACK[]` tails.

## Result (EuRoC MH_03_medium, 2700 frames)

SuperPoint (2048 kpts) + LightGlue stereo/temporal matching, PnP relative pose,
30-iteration global BA.

| Metric | Value |
| --- | ---: |
| Merged multi-view tracks (landmarks) | **178,973** |
| Total observations | **2,029,024** |
| Mean track length | **11.3 frames** (median 5, max 615) |
| **Mean reprojection error, before BA** | 4.08 px |
| **Mean reprojection error, after BA** | **1.04 px (3.9×)** |

The reprojection error is the headline: a per-frame lift sits at ~4 px of
multi-view disagreement; one global BA over the merged tracks tightens it to
~1 px — the multi-view consistency a single stereo lift cannot provide and the
form COLMAP feeds to 3DGS.

### On the downstream 3DGS quality

Whether the reconstruction renders **crisp** in 3DGS turns out to depend almost
entirely on the **capture trajectory**, not on visloc's pose precision:

| Sequence | Motion | Reprojection (after BA) | 3DGS render |
| --- | --- | ---: | --- |
| **MH_03** (Machine Hall) | forward fly-through | 1.04 px | recognisable but **blurry** |
| **V2_03** (Vicon Room) | orbit-like, 150-frame window | **0.53 px** | **crisp** (see README) |

On MH_03 the drone flies *forward through* the hall: each surface is seen from a
narrow range of angles, so 3DGS cannot pin its depth and the gaussians smear —
the same softness any pose source (COLMAP included) would produce on that
trajectory. We confirmed this is *not* fixable downstream: per-image exposure
compensation, longer training, more gaussians, and MCMC scale/opacity
regularisation all leave it blurry.

On V2_03 the flight **orbits** the room, giving each surface rich multi-view
parallax. The same SfM pipeline then tightens to **sub-pixel (0.53 px)**
reprojection, and the resulting splat — rendered from visloc-rs's *own* estimated
poses — is sharp enough to be nearly indistinguishable from the real photo. This
is the proof that visloc's poses are reconstruction/NVS-grade when the capture
suits 3DGS; the MH_03 blur was capture geometry, not pose error.

## Reproduce

```sh
scripts/run_euroc_sfm_benchmark.sh --mav0 /path/to/MH_03_medium/mav0 --frames 2700
```

Reuses the same rectified images and SuperPoint/LightGlue features as the
loop-closure benchmark, so pass `--rect-dir` / `--feat-dir` to skip re-deriving
them. The COLMAP model lands in `target/euroc_sfm_benchmark/colmap`. To render
it as a 3D Gaussian Splat, place the rectified left images under
`<base>/undistorted/images`, the COLMAP `*.txt` under `<base>/undistorted/sparse`,
and run `scripts/gsplat_mcmc_train.py <base> <out.splat> 7000 400000`.

Requires python with `torch` + `lightglue` + `opencv` for the rectify/export
stages and the `stereo_vo_external_deep_files` example.

For the **crisp orbit splat** (the README figure), run the same flow on an
orbit-style capture. We used EuRoC V2_03 (Vicon Room 2), frames 0–150 (it is a
"difficult" sequence with a later sensor blackout, so the window stops before
it): `--features-dir <V2_03 features> --frames 150 --width 752 --height 480
--sfm-colmap-out <dir>` gives 0.53 px reprojection, and
`scripts/gsplat_mcmc_train.py <base> <out.splat> 7000 300000` renders the crisp
result.
