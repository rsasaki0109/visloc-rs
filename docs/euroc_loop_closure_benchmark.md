# EuRoC Loop-Closure Benchmark (UAV, 6-DOF)

The aerial counterpart to the [KITTI loop-closure
benchmark](kitti_loop_closure_benchmark.md). KITTI is a ground vehicle on a
plane; EuRoC MH_03 is a **micro aerial vehicle on a 6-DOF flight** through a
GPS-denied machine hall with Vicon/Leica ground truth. The same comparison
holds: an open stereo-VO trajectory accumulates unbounded drift, and a loop
closure — the one constraint that ties a revisited place back to its earlier
observation — removes it.

The pipeline is identical to KITTI's
(`pipelines/slam/src/vo_loop_closure.rs`: VLAD appearance → PnP metric
verification → robust GNC SE(3) pose-graph optimization), run over the same
file-backed SuperPoint/LightGlue stereo features twice (open vs
`--loop-closure`). Only the front matter differs:

- **Rectification.** EuRoC's cam0/cam1 are radtan-distorted (unlike KITTI's
  pre-rectified odometry images), so `scripts/rectify_euroc_stereo.py`
  undistorts+rectifies to a pinhole stereo pair and writes a KITTI-format
  `calib.txt` + a `timestamps.txt`.
- **Timestamped ground truth.** EuRoC GT is a ~200 Hz timestamped trajectory,
  so VO poses are converted to TUM (`scripts/kitti_poses_to_tum.py`, using the
  rectifier's `timestamps.txt`) and scored with `evo_ape`, which associates by
  timestamp.

## Result (EuRoC MH_03_medium, 2700 frames)

SuperPoint (2048 kpts) + LightGlue stereo/temporal matching, PnP relative pose.
ATE rmse via `evo_ape` under SE(3) (rigid, the metric-frame number) and Sim(3)
(scale-absorbed) alignment.

| trajectory     | ATE rmse SE(3) | ATE rmse Sim(3) |
| -------------- | -------------: | --------------: |
| open VO        |        2.462 m |         2.203 m |
| + loop closure |        0.464 m |         0.443 m |

**5.3× lower SE(3) ATE rmse (2.46 m → 0.46 m)** from 307 verified loops (5051
appearance candidates), GNC pose-graph cost `1.05e6 → 230`. Unlike the KITTI
stride-2 contrast, **both alignments improve cleanly** (SE(3) 5.3×, Sim(3) 5.0×)
because the metric stereo frontend keeps the scale correct here — the open VO
integrates to 123.9 m against the true ~131 m path — so the metric PnP loop
edges and the odometry edges agree and the pose graph closes without distorting
the metric frame.

![EuRoC MH_03 open VO vs loop closure](assets/euroc_mh03_loop_closure.png)

The open VO (left, red) drifts off ground truth over the flight; loop closure
(right, green) pulls the revisited places back onto the Vicon/Leica trajectory.

## Closing the gap to state of the art: window BA + loop closure

The table above deliberately isolates the loop-closure *stage* by running it on
an **open** VO (no bundle adjustment), so the only variable is the loop closure.
The full pipeline visloc actually ships pairs that same loop closure with an
**incremental windowed bundle-adjustment frontend** (`--online-ba`, a 30-frame
sliding-window local BA triggered every 10 frames). That combination is far
closer to state of the art — measured on the same artifacts, two EuRoC machine
hall flights:

| pipeline                                       | MH_03 ATE SE(3) | MH_05 ATE SE(3) |
| ---------------------------------------------- | --------------: | --------------: |
| open VO, no BA, no loop                        |         2.462 m |         2.387 m |
| open VO + loop (stage isolation, above)        |         0.464 m |               — |
| window BA + loop                               |         0.089 m |         0.119 m |
| window BA + loop + two-view loop BA            |         0.065 m |         0.086 m |
| + fixed-prefix local-map BA (history 20)       |         0.061 m |         0.084 m |
| + anisotropic loop-edge information            |         0.060 m |         0.080 m |
| **+ BA track init-residual gate (10 px)**      |       **0.057 m** |       **0.072 m** |

The final rung (`--ba-max-init-residual 10`) gates each BA track by its
initial reprojection residual against the RANSAC-validated frontend poses.
It was found on KITTI (it filters dynamic-object tracks that bundle
adjustment would otherwise trust — BA track building has no RANSAC; see the
[KITTI multi-sequence benchmark](kitti_multiseq_benchmark.md)) and turns out
to also help both EuRoC flights, where it trims the noisiest depth-lift
tracks.

Both alignments agree (MH_03 Sim(3) 0.046 m, MH_05 Sim(3) 0.064 m), so the metric
scale is correct, not absorbed by the fit. Against the published deep/classical
stereo SLAM systems:

| system (stereo)                       | MH_03 ATE rmse | MH_05 ATE rmse |
| ------------------------------------- | -------------: | -------------: |
| ORB-SLAM3 (Campos et al. 2021)        |        0.024 m |        0.052 m |
| DROID-SLAM (Teed & Deng 2021)         |        0.035 m |        0.040 m |
| OV2SLAM (Ferrera et al. 2021)         |         0.04 m |         0.07 m |
| VINS-Fusion stereo (Qin et al. 2019)  |         0.33 m |         0.50 m |
| **visloc-rs (full pipeline)**         |     **0.057 m** |     **0.072 m** |

visloc-rs lands within **~2.4× of ORB-SLAM3** and **~1.6× of DROID-SLAM** on
MH_03 and within **~1.4× of ORB-SLAM3** on MH_05 — at OV2SLAM's real-time
accuracy and 5–7× ahead of VINS-Fusion's stereo-only mode — in pure Rust.

### Two-view loop bundle adjustment

The `--loop-two-view-ba` stage is what takes the pipeline from 0.089 m to
0.065 m (MH_03) and 0.119 m to 0.086 m (MH_05) — a consistent **~28 %** ATE
reduction on both flights, from the *same* 317 / 282 verified loops. Each loop
edge enters the pose graph from a PnP estimate that minimises reprojection in the
**newer** frame only, while holding the **older** frame's stereo-depth points
fixed — so any error in the older disparity triangulation passes straight into
the edge. The refinement re-grinds each loop as a minimal two-view bundle
adjustment in the older camera frame: the older pose is the fixed gauge, the newer
pose and the shared landmarks are free, and the older rectified-stereo disparity
becomes a *soft* metric anchor (the points may slide off it to satisfy
reprojection in **both** views instead of being frozen at the noisy depth). The
metric scale stays well-posed because the older stereo residuals pin it. The PGO
cost drops sharply (MH_03 213 → 25, MH_05 104 → 9): the refined edges are
mutually consistent in a way the depth-biased PnP edges were not.

Crucially this is **local** — it touches only each loop's two frames, never the
rest of the trajectory; the global drift distribution stays with the SE(3) PGO.
A *global* post-loop bundle adjustment (merging the loop correspondences into the
full track set and re-optimising the whole flight) was tried and **discarded**:
it deforms the already locally-consistent window-BA trajectory and *worsens* ATE
(0.089 → 0.15 m), the same way a loop-free global reprojection-minimum does. The
lever is grinding each loop edge well, not re-solving everything.

### Fixed-prefix local-map bundle adjustment

The last stage (`--online-ba-history 20`) is what takes the pipeline from 0.065 m
to **0.061 m** (MH_03) and 0.086 m to **0.084 m** (MH_05) — Sim(3) ATE drops
~10 % / ~6 %. It is the ORB-SLAM3 *fixed-keyframe local BA* pattern brought onto
the streaming window: each `--online-ba` trigger extends its optimisation window
**backward** by 20 frames and holds that older prefix **fixed**, so landmarks
established in earlier windows anchor the recent poses over a baseline far longer
than the 30-frame window — without re-optimising (and so without deforming) the
already-settled older trajectory.

This sharply distinguishes it from naively *widening* the window: a 50/60/80-frame
window with every pose free *worsens* the result (the old poses get re-perturbed by
stale, off-view observations), whereas the same extension with the prefix fixed
*improves* it. The fixed anchor is the whole point. The history length has a sweet
spot — 20 frames helps both flights, but a longer history (30+) starts admitting
stale landmarks and on MH_05 turns slightly negative, so 20 is the shipped default.

### Anisotropic loop-edge information

The `--loop-edge-information` stage takes MH_03 Sim(3) ATE from 0.053 m to
**0.052 m** and MH_05 from 0.084 m / 0.072 m (SE3 / Sim3) to **0.080 m / 0.069 m**
(~5 % on both alignments) — from the *same* verified loops. It is the ORB-SLAM
*Essential-Graph per-edge information* that the pose graph had been omitting.

Every loop edge entered the SE(3) PGO with a single isotropic weight (the inlier
count), pulling all six DOF equally. But a loop edge is a PnP / two-view-BA
estimate: it constrains rotation and the two lateral image directions tightly and
the optical-axis (depth) direction weakly. With an isotropic weight the solver
smears each loop correction uniformly over the cycle; with the edge's true
ellipsoidal information it routes the correction into the directions the loop
actually observes. `loop_edge_information` recovers `Ω` as the reprojection Hessian
`Σ JᵀJ` of the loop measurement — finite-differenced in the solver's own
`[ρ; ω]` SE(3) right-perturbation tangent, so there is no convention mismatch — and
**trace-normalises it to the same total weight `inlier_count`**. That isolates the
*direction* of each loop's pull as the only changed variable; the calibrated
loop-vs-odometry magnitude (which the earlier stages tuned) is preserved. The
effect on consistency is stark: the GNC pose-graph cost collapses (MH_05 final cost
9.3 → 0.19, MH_03 24.8 → 0.43) because the anisotropically-weighted loops are
mutually consistent in a way the isotropic edges were not.

The remaining gap to ORB-SLAM3 narrows but does not close. Its local BA selects
the true **covisibility graph** (every keyframe sharing a landmark observation),
while ours is still a *temporal* window — a good proxy on a mostly-forward flight,
but it never pulls in a spatially-near non-consecutive keyframe. Two measured
caveats remain: the windowed BA must be the *streaming* `--online-ba` (a one-shot
full-batch `--final-global-ba` over the whole flight is both far slower and worse,
0.214 m, for the same deform-the-trajectory reason); and lifting the
loop-verification cap (which admits low-similarity false loops) *worsens* the
result.

## The frame-gap gate matters more in the air

EuRoC flies at 20 Hz, and the MAV spends time hovering/slow-maneuvering. A small
`--loop-min-frame-gap` then matches only slow-motion temporal neighbours that
are already odometry-consistent and contribute no drift correction. Measured on
MH_03:

| `--loop-min-frame-gap` | meaning  | ATE rmse SE(3) |
| ---------------------: | -------- | -------------: |
| 30                     | ~1.5 s   |     2.46 m (unchanged) |
| 200                    | ~10 s    |     0.46 m |

A loop only corrects drift if enough drift accumulated between its two frames,
and drift grows with *distance travelled*, not frame index — so the path-length
gate (`--loop-min-path-length`, metres) is the frame-rate-independent version of
the same idea. The default is `--loop-min-frame-gap 200`.

## The full EuRoC suite: where a vision-only stereo stack stands

The sections above focus on the two machine-hall flights. Running the *same*
pipeline over all eleven EuRoC sequences draws a clear and honest line: this is
a **vision-only** stereo SLAM stack (no IMU in the estimator), and EuRoC is the
benchmark that was built to show why micro-aerial vehicles need inertial
sensing. The machine-hall flights are large, well-lit, and forward-moving — a
regime where metric stereo plus a deep frontend plus windowed BA is genuinely
competitive. The Vicon-room flights are small, fast-rotating, and motion-blurred
— the regime the IMU was added for.

ATE RMSE (m), `evo_ape`, timestamp-associated to the Vicon/Leica ground truth,
SE(3) alignment. visloc runs the headline config with `--min-depth 0.5` (see the
depth note below); ORB-SLAM3 columns are the published Table II medians
(arXiv:2007.11898).

| seq   | visloc SE(3) | visloc Sim(3) | ORB-SLAM3 stereo-inertial | ORB-SLAM3 stereo |
| ----- | -----------: | ------------: | ------------------------: | ---------------: |
| MH_01 |     0.052 m  |     0.048 m   |                  0.036 m  |        0.029 m   |
| MH_02 |     0.054 m  |     0.040 m   |                  0.033 m  |        0.019 m   |
| MH_03 |     0.054 m  |     0.052 m   |                  0.035 m  |        0.024 m   |
| MH_04 |  0.106 m †   |     0.098 m   |                  0.051 m  |        0.085 m   |
| MH_05 |   **0.065 m**|     0.060 m   |                  0.082 m  |        0.052 m   |
| V1_01 |     0.091 m  |     0.090 m   |                  0.038 m  |        0.035 m   |
| V1_02 |     0.072 m  |     0.070 m   |                  0.014 m  |        0.025 m   |
| V1_03 |     0.161 m  |     0.161 m   |                  0.024 m  |        0.061 m   |
| V2_01 |     0.180 m  |     0.179 m   |                  0.032 m  |        0.041 m   |
| V2_02 |     0.129 m  |     0.127 m   |                  0.014 m  |        0.028 m   |
| V2_03 |   **DNF**    |       —       |                  0.024 m  |        0.521 m   |

What the table says, read honestly:

- **Machine hall is competitive.** visloc lands within ~1.5× of ORB-SLAM3's
  *stereo-inertial* numbers on MH_01/02/03, **beats** its stereo-inertial result
  on **MH_05 (0.065 vs 0.082 m)**, and beats its stereo-vision result on MH_04
  (vs 0.085 m). This is OV2SLAM-class accuracy, in pure Rust, without an IMU in
  the estimator.
- **The Vicon rooms are the vision-only ceiling.** On V1_03/V2_01/V2_02 visloc
  is 2.5–11× behind ORB-SLAM3 stereo-inertial. These are the sequences whose
  ATE is dominated by fast rotations and motion blur, where the IMU's rotation
  rate and gravity direction are the load-bearing measurements. Even
  ORB-SLAM3's *stereo-vision* mode degrades here (V2_03 = 0.521 m, an effective
  failure); a vision-only stack does not have the signal these sequences need.
- **V2_03 does not complete.** It contains a genuine sensor blackout. The
  `--min-depth` fix carried the frontend from its old death at pair 206 to pair
  479, but the blackout still starves the stereo frontend (its triangulated-pair
  count collapses 59 → 5) and the relative pose fails. Bridging it requires
  tightly-coupled inertial propagation, which this stack does not have.

### The depth gate is scene-scale dependent

EuRoC's two scene scales want different `--min-depth` values, and no single
value is best for all eleven:

- The machine hall's default 3.0 m gate (tuned for the parallax-noise floor at
  vehicle/hall scale) makes **MH_02 crash** — it has a close-approach segment
  (~0.64 m) the gate rejects into starvation — so MH_02 needs ~0.5 m.
- The Vicon rooms (median stereo depth ~2 m) reject nearly all points at 3.0 m
  and every one of them crashes; they need ~0.5 m.
- But MH_04 is *worse* at 0.5 m (0.106 m) than at 3.0 m (**0.060 m**, the † row)
  — the looser gate admits near-field noise its longer baselines would
  otherwise outweigh.

A single global `--min-depth` is therefore a compromise; a frame-adaptive depth
gate (keyed on the running median triangulated depth) is the clean fix and is
noted as future work. The machine-hall headline numbers elsewhere in this doc
use the 3.0 m default; this suite uses 0.5 m so the Vicon rooms run at all.

## Reproduce

Fetch the EuRoC `MH_03_medium` ASL-format zip (the one with a `mav0/` folder) —
EuRoC is hosted on the ETH Research Collection — then:

```sh
scripts/run_euroc_loop_closure_benchmark.sh \
  --mav0 /path/to/MH_03_medium/mav0 \
  --frames 2700
```

The script rectifies the stereo pair, exports SuperPoint/LightGlue features once
(needs torch+lightglue+opencv), runs the VO twice (open / `--loop-closure`),
derives the GT TUM from `mav0/state_groundtruth_estimate0`, converts both VO
trajectories to TUM, scores them with `evo_ape` (needs `pip install evo`), and
writes `summary.md` with the table above. `--rect-dir`/`--feat-dir`/`--gt-tum`
reuse already-prepared artifacts; `--device cpu` runs the export without CUDA.
