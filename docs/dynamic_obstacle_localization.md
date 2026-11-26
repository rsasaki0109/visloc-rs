# Map localization among moving people: measurements

Results page of [dynamic_obstacle_localization_plan.md](dynamic_obstacle_localization_plan.md).
This is Phase 0: today's localization, measured indoors with people walking
in front of the camera, before any lever. Nothing in the localization has
changed; only measurement was added.

**Headline.** With people walking through the view (TUM RGB-D fr3 walking),
single-image localization accepts many wrong poses: **22 to 145 fixes more
than 10 cm off** per sequence, none on the sequence where they sit still.
Most wrong fixes have their inliers bunched in a few image cells: a fix with
inliers in at most 4 of 16 cells is more than 10 cm off **56%** of the
time, one with 8 or more only **5%**. With an odometry, the correction's
gate keeps the map pose within 2–4 cm on three of the four walking
sequences, but still takes in 3–27 wrong fixes, and on walking_rpy the map
pose is 10 cm RMSE (20 cm worst). The walking people hide a third of the
image cells where the map has landmarks (matched in 58–67% of them, against
92% with the people seated).

## What was added (measurement only)

- `visloc_localization::{ExpectedVisibility, ImageGrid, view_diagnostics}`
  and `basalt_euroc_localize_demo --diagnostics FILE [--diag-grid N]
  [--eval-mask-dir DIR]`: per attempt,
  - `inlier_cells`: cells of a 4 × 4 image grid holding an inlier;
  - `expected`: the landmarks the map's 5 nearest keyframes (within 1 m and
    30° of the camera's axis) saw, projected into the image, from the
    predicted pose (with an odometry) or the solved one;
  - `expected_cells` / `missing_cells`: cells holding 3 or more of them,
    and of those the cells holding no inlier at all;
  - with masks: keypoints and inliers on moving things.
  Every other output is byte-identical with and without `--diagnostics`
  (checked on walking_xyz in both modes; the latency column aside). Cost:
  0.1 ms median per attempt (0.3 ms worst), against 18–60 ms of
  localization.
- Landmark counts are of limited use on these maps: a map built along a
  sequence keeps a landmark per stretch of keyframes that tracked a point,
  so one physical point has several. The cell counts do not care, and are
  what the tables use.
- `scripts/tum_rgbd_to_euroc.py`, `scripts/synth_drift_odometry.py`,
  `scripts/eval_dynamic_localization.py`.

## Setup

- **Data.** TUM RGB-D Freiburg 3, from `cvg.cit.tum.de` (CC BY 4.0):
  `sitting_xyz` and `sitting_halfsphere` (two people seated at a desk,
  moving a little; handheld camera moving) for the map; queries
  `sitting_static` (people seated, camera still) as the control and
  `walking_static`, `walking_xyz`, `walking_halfsphere`, `walking_rpy`
  (two people walking around the desk and in front of the camera). All were
  recorded in one session in one motion-capture frame. SHA-256 of the
  archives:

  | Archive | SHA-256 |
  | --- | --- |
  | `rgbd_dataset_freiburg3_walking_xyz.tgz` | `1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d` |
  | `rgbd_dataset_freiburg3_walking_static.tgz` | `8aca832bf01746aa95fcc0b15bf42d78ad3bb8edef672ba4cc94a404b194f6c1` |
  | `rgbd_dataset_freiburg3_walking_halfsphere.tgz` | `c4984f4894724398590dcf18c3d9d54dbb76cce925d51695f44ad8e697fc8458` |
  | `rgbd_dataset_freiburg3_walking_rpy.tgz` | `8727e72a01da0150996aa98a63682e1798e4d6d196c30d35e8f693b33fba3d5d` |
  | `rgbd_dataset_freiburg3_sitting_static.tgz` | `6239d4b7f6edcab719b51c76434acdc238d03dad666758b51e6046992a7c3914` |
  | `rgbd_dataset_freiburg3_sitting_xyz.tgz` | `05c071672cda22a668860a935124737a4eb4fa772cbad372e73d5a99ce4be205` |
  | `rgbd_dataset_freiburg3_sitting_halfsphere.tgz` | `ba9f0fab0d07e22f04fbfae16eb4e3fb44088a32c920ad36c782b5024ed4b767` |

- **Map.** `tum_rgbd_save_map --keep-world` along the ground truth of
  `sitting_xyz` and `sitting_halfsphere` concatenated (their stamps do not
  overlap): 663 keyframe images, 40,242 landmarks. The map is in the
  ground truth's frame, so poses are scored directly, with nothing fitted.
  The seated people are in it (F3 is part of what is measured).
- **Why not a static-camera map.** A map from `sitting_static` alone (the
  camera never moves) localized 0–6 images of each walking sequence: their
  views start 25 cm or more away, and that map has one viewpoint.
- **Query settings.** `--ratio none --max-hamming 60 --min-inliers 20`.
  With the default ratio test (tuned on EuRoC), even `sitting_static`
  localized only 139 of 707 images against a `sitting_xyz` map; without it,
  707. The many near-duplicate landmarks of an RGB-D map along a sequence
  make the second-best distance close to the best.
- **Modes.**
  - *Single image*: every image localized on its own. Shows the raw fixes.
  - *Odometry*: `--odometry` with a drifting odometry made from the ground
    truth (`synth_drift_odometry.py`: 2% scale, 3°/m of yaw, step noise;
    roll and pitch kept, as an IMU would), `--localize-every 3` (10 Hz).
    Freiburg 3 records no IMU. Three seeds.
- **Machine.** 4-core container, release build, `--threads 4`.

## Single image

| Query | Localized | Longest gap | Fix error RMSE / median / max | Fixes > 10 cm | Inlier cells (median) | Mapped cells matched (median) | Time / image |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| sitting_static (control) | 707/707 (100%) | 0.1 s | 2.9 / 2.2 / 8.5 cm | **0** | 12 | 92% | 50 ms |
| walking_static | 730/743 (98.3%) | 0.3 s | 4.2 / 2.8 / 25.3 cm | **22** | 8 | 67% | 57 ms |
| walking_xyz | 710/859 (82.7%) | 1.2 s | 7.1 / 3.4 / 59.1 cm | **70** | 7 | 62% | 60 ms |
| walking_halfsphere | 602/1067 (56.4%) | 3.4 s | 10.3 / 4.5 / 84.8 cm | **95** | 8 | 67% | 58 ms |
| walking_rpy | 385/910 (42.3%) | 2.7 s | 16.2 / 7.0 / 129.2 cm | **145** | 7 | 67% | 57 ms |

The runs are deterministic (the same output bytes on a rerun).

**Inlier spread against error**, all accepted fixes of the four walking
sequences:

| Inlier cells (of 16) | Fixes | > 10 cm off |
| --- | ---: | ---: |
| ≤ 4 | 93 | 52 (**56%**) |
| 5–6 | 540 | 145 (27%) |
| 7 | 483 | 67 (14%) |
| ≥ 8 | 1,295 | 68 (**5%**) |

On `sitting_static`, every fix has 5 or more inlier cells (median 12) and
none is 10 cm off.

![The four worst fixes of walking_xyz (59, 49, 46 and 38 cm off) and two good ones (1 cm), with their inliers in red. The bad ones have their inliers on the posters on the partition, bunched in one or two places; the good ones have them over the desk, the monitors and the partitions.](assets/dynamic_tum_wrong_fixes.jpg)

The four worst fixes of walking_xyz (top row and bottom left: 59, 49, 46,
38 cm) and two good ones (1 cm), inliers in red. The bad ones are not on
the people: their inliers sit on the posters of the partition, in one or two
blobs, seen from farther back than the map was. The good ones spread over
desk, monitors and partitions. Images: TUM RGB-D, CC BY 4.0.

## With an odometry (three seeds)

Median of the three seeds, range in brackets.

| Query | Images localized | Map pose RMSE | Map pose max | Wrong fixes used (> 10 cm) | Resets |
| --- | --- | --- | --- | --- | --- |
| sitting_static (control) | 100% | 1.5 (1.3–1.8) cm | 2.8 (2.4–3.3) cm | 0 | 0 |
| walking_static | 98.8% | 2.2 (1.8–2.3) cm | 4.0 (2.7–4.4) cm | 3 (2–4) | 0 |
| walking_xyz | 87.8% | 2.0 (1.7–4.8) cm | 5.9 (3.7–12.9) cm | 12 (11–21) | 1 (0–1) |
| walking_halfsphere | 67.7% | 3.1 (2.8–4.4) cm | 9.2 (9.2–10.4) cm | 13 (11–16) | 3 (2–3) |
| walking_rpy | 55.6% | **10.0** (10.0–10.1) cm | **20.2** (19.7–20.6) cm | **27** (23–27) | 1 |

"Wrong fixes used" are fixes more than 10 cm off that the correction
accepted, initialized or reset from. Its gate (15 cm and 3°, widened with
travel) lets through fixes that are wrong by less than the gate, and they
are averaged into the correction.

## Failure modes (F1–F5 of the plan)

- **F2/F5, bunched inliers: the main source of wrong fixes.** Wrong fixes
  have their inliers in few cells, on one textured patch (posters), often
  from a view farther back than the map's. The people take away the rest
  of the view, so what is left is a small, nearly planar patch, and P3P
  RANSAC on it is poorly conditioned. Movers' own keypoints are rarely the
  inliers in these images. Lever D1 (inlier spread) addresses this
  directly.
- **F1, occlusion.** The walking people hide about a third of the mapped
  cells (58–67% matched against 92%). Fewer images localize (42–98%), but
  the longest gap without a fix stays at 3.7 s or less.
- **F5 with an odometry.** 3–27 wrong fixes pass the gate per sequence; on
  walking_rpy (fast rotation) they hold the map pose 10 cm off.
- **F3, people in the map.** The map holds the seated people. Not separated
  out yet: no masks for TUM.
- **F4, dirty odometry.** Not tested: the odometry here is synthetic.

## Phase 1, first levers: fix quality and the correction's start

Two opt-in switches in `basalt_euroc_localize_demo`, both off by default
(outputs unchanged with them off, checked on walking_xyz):

- `--min-inlier-cells K` (lever D1, `visloc_localization::apply_inlier_spread_gate`):
  a localization whose inliers occupy fewer than K cells of the 4 × 4 grid
  is turned down as if it had too few inliers (it can still be a weak fix).
- `--min-inlier-ratio R`: the same below an inlier ratio R
  (`LocalizationConfig::min_inlier_ratio`, now exposed).
- `--init-fixes N`, `--reset-after N`: `OdomCorrectionConfig`'s agreeing
  fixes that start the correction (2) and that replace it (3).
- `--init-min-inliers N`: until the correction is set, only fixes with at
  least N inliers help start it (the others are logged as `held-back`).

### Single image: inlier cells are a weaker test than the inlier ratio

Wrong fixes are the fixes more than 10 cm off; the four walking sequences
together have 332 wrong and 2,095 good fixes without a gate.

| Gate | Wrong fixes | Good fixes |
| --- | ---: | ---: |
| none | 332 | 2,095 |
| `--min-inlier-cells 5` | 280 (−16%) | −2% |
| `--min-inlier-cells 6` | 213 (−36%) | −9% |
| `--min-inlier-cells 7` | 135 (−59%) | −21% |
| `--min-inlier-cells 8` | 68 (−80%) | −41% |

Ranking the same fixes by each feature, at the threshold that loses 10% of
the good fixes:

| Feature | Wrong fixes removed at 10% good lost |
| --- | ---: |
| inlier ratio (≥ 0.165) | 63% |
| inlier count (≥ 32) | 58% |
| inliers' convex hull (≥ 13% of the image) | 46% |
| inlier cells (≥ 6) | 36% |
| mapped cells without an inlier | 18% |

A grid search over all three together does no better than the inlier
ratio alone (61% at 9.4% lost). The thresholds were picked on the same four
sequences they score, so these numbers are optimistic. **The cell count,
the lever the plan proposed, is the weakest of the three**: the wrong fixes
bunch their inliers, but they also have few and a low share of them, and
those two say it better.

### With an odometry: the gates help walking_xyz, not walking_rpy

Median of three seeds (range in brackets); the control (sitting_static) is
unchanged by every setting (1.5–1.6 cm, no wrong fix used).

| Setting | walking_xyz RMSE / wrong used | walking_halfsphere RMSE / wrong used | walking_rpy RMSE / wrong used |
| --- | --- | --- | --- |
| none | 2.0 (1.7–4.8) cm / 12 | 3.1 (2.8–4.4) cm / 13 | 10.0 cm / 27 |
| `--min-inlier-cells 6` | 1.7 (1.6–1.9) cm / 6 | 2.8 (2.7–4.1) cm / 11 | 9.5 cm / 16 |
| `--min-inlier-cells 7` | 1.7 (1.5–1.8) cm / 3 | 3.1 (2.7–3.4) cm / 9 | 11.1 cm / 9 |
| `--min-inlier-ratio 0.16` | 2.0 (1.6–4.8) cm / 11 | 3.0 (2.8–3.6) cm / 10 | 10.9 cm / 19 |
| `--init-fixes 4` | 2.5 (2.2–4.0) cm / 12 | 3.0 (2.7–3.6) cm / 11 | 7.8 cm / 27 |
| `--init-fixes 3 --reset-after 2` | 3.2 (1.6–5.0) cm / 10 | 3.7 (3.4–5.2) cm / 15 | 7.5 (5.8–7.8) cm / 24 |
| `--init-fixes 4 --min-inlier-cells 6` | 1.7 (1.7–1.9) cm / 6 | 2.8 (2.7–3.4) cm / 9 | 8.9 cm / 16 |

With the projection search, the inlier ratio is high for most images, so
the ratio gate does little here. The cell gate halves the wrong fixes used
on walking_xyz and brings its worst seed from 4.8 to 1.9 cm, at the cost of
fewer localized images (88% → 76% with 6 cells, 59% with 7). None reaches
the target of no wrong fix used.

**walking_rpy's 10 cm is its start, not stray wrong fixes.** In the first 2 s
every fix is 11–25 cm off in the same direction (21–29 inliers, a view of
the partition with people in front), so they agree with each other and set
the correction; the good fixes that follow (1–3 cm) disagree with it and
are rejected, mixed with more wrong ones, until a reset at 7 s. From 8 s on
the map pose stays at 2.5–5.7 cm (median per 2 s). A rule that trusts
agreeing fixes cannot tell a consistent bias from the truth, so neither
more initial fixes nor a quicker reset fixes it (7.5–7.8 cm at best, and a
quicker reset hurts the other sequences).

`--fusion` (the sliding-window `MapFusion`) does not help either: it is
fed the fixes the correction used, so it inherits the wrong start (walking_rpy
12.1 cm).

### Starting the correction only from strong fixes

The biased first fixes have 21–29 inliers; the good ones later mostly 40–70.
Asking more inliers of the fixes that start the correction than of the fixes
that keep it going (`--init-min-inliers`) removes the bad start:

| Setting | sitting_static | walking_static | walking_xyz | walking_halfsphere | walking_rpy | walking_rpy max | Frames with a map pose (rpy) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1.5 | 2.2 | 2.0 (1.7–4.8) | 3.1 (2.8–4.4) | 10.0 | 20.2 | 901 of 906 |
| `--init-min-inliers 35` | 1.5 | 2.2 | 2.0 (1.7–4.8) | 2.9 (2.7–3.7) | **3.2** (3.1–3.7) | **7.0** | 802 |
| `--init-min-inliers 45` | 1.5 | 2.2 | 2.0 (1.7–4.8) | 2.8 (2.6–3.4) | 3.5 (3.0–4.6) | 10.6 | 794 |
| `--init-min-inliers 35 --min-inlier-cells 6` | 1.5 | 2.2 | **1.7** (1.6–1.9) | 2.8 (2.6–3.5) | 3.7 (3.3–4.8) | 8.0 | 802 |

Map pose RMSE in cm, median of three seeds (range). The correction starts
3.3 s later on walking_rpy (the first 99 frames get no map pose) and 0.5 s
later on walking_halfsphere; the other sequences start as before. The
wrong fixes the correction then uses (9–26 per sequence) are mostly 10–15 cm
off, inside its gate, and are averaged with good ones; they no longer
move the pose much.

### Where this leaves Phase 1

- **walking_rpy's target (under 5 cm) is met** with `--init-min-inliers 35`,
  without hurting the others. The other target, no wrong fix used, is not:
  the gate (15 cm) is wider than the threshold that counts a fix as wrong
  (10 cm).
- Best combination measured: `--init-min-inliers 35 --min-inlier-cells 6`,
  1.5–3.7 cm on all five sequences (worst seed 4.8 cm), control unchanged,
  at the cost of fewer localized images (walking_xyz 88% → 76%).
- These were tuned on the sequences they are scored on. No default
  changes until they hold on another indoor set (Bonn, OpenLORIS cafe).
- Without an odometry, the inlier ratio is the better single-image test.

## Bonn RGB-D Dynamic: a second indoor set

**Result.** On Bonn, today's localization finds few fixes (0–33% of the
images of the dynamic sequences, 0.3% of a static one from another day),
and the setting that fixed TUM walking_rpy (`--init-min-inliers 35`) does
not carry over: fixes reach 35 inliers so rarely that the correction
starts late or never. **No default changes**, and the TUM thresholds stay
tuned to TUM.

### Ground truth of the camera

Bonn's `groundtruth.txt` is the pose of the motion-capture markers, and the
published marker-to-camera matrix is not a rotation. `scripts/rgbd_hand_eye.py`
calibrates `T_body_cam` from the data: ORB matches between frame pairs with
the registered depth give the camera's relative motion (RANSAC rigid fit),
the ground truth gives the markers', and AX = XB is solved (Park–Martin
rotation, least-squares translation), with the time offset searched on a
grid.

- Pooling `static` (2019-01-23) and `static_close_far` (2019-01-24), 221
  pairs: `T_body_cam` = `-0.00933,0.10053,-0.02389,-0.125559,0.882537,-0.450645,0.047777`
  (tx,ty,tz,qx,qy,qz,qw), time offset 0–10 ms, residual 0.56° / 1.6 cm per
  pair (median).
- Each sequence alone gives the same rotation (within 1°) but translations
  8 cm apart: walking with a handheld camera turns it mostly about the
  vertical, which leaves part of the translation weakly observed. Pooled
  with pairs 8 or 20 frames apart, the translation moves by under 1 cm.
- The accepted fixes on the dynamic sequences are off by a consistent 4–5
  cm (mean error vector), which is the level of this calibration: errors
  under about 5 cm cannot be told apart on Bonn, so fixes count as wrong
  from 15 cm here.

### Map and setup

- `static` has depth images for its first 337 s only (the archive holds
  10,000 of the 10,908 listed in `depth.txt`; it unzipped without error),
  so the map uses that
  stretch, plus `static_close_far`: `tum_rgbd_save_map --keep-world
  --match-radius-px 8 --track-keyframes 10 --keyframe-m 0.05 --keyframe-deg 5`
  with the published intrinsics and distortion, 1,987 keyframe images,
  108,963 landmarks.
- With the defaults (4 px, 5 keyframes) the map kept 23 landmarks per
  keyframe and localized only 52% of its own images; tracking over more
  keyframes with a wider radius brings that to 82% (1.7 cm median).
- Queries: `crowd`, `crowd2` (three people walking around), `person_tracking`
  (the camera follows a walking person), `moving_obstructing_box`,
  `removing_obstructing_box` (a person carries a box through the view or
  takes it away), with the TUM query settings.

### Single image

| Query | Localized | Fix error RMSE / median / max | Fixes > 15 cm | Inlier cells (median) |
| --- | --- | --- | ---: | ---: |
| crowd | 0/928 | — | — | — |
| crowd2 | 0/895 | — | — | — |
| person_tracking | 90/580 (15.5%) | 10.5 / 8.9 / 20.7 cm | 9 | 9 |
| moving_obstructing_box | 108/590 (18.3%) | 8.7 / 6.9 / 18.9 cm | 3 | 9 |
| removing_obstructing_box | 284/963 (29.5%) | 9.0 / 7.2 / 25.3 cm | 14 | 10 |

Most images match the map in many places (150–180 matches) but few
matches agree (median 4–12 inliers). The map's nearest views are 0.4–1 m
from the queries' and look at the white-walled room from other angles; the
`crowd` sequences stand where the mapping walk never stood. That is map
coverage, not the people.

### With an odometry (three seeds; fixes > 15 cm count as wrong)

| Query | Setting | Images localized | Frames with a map pose | Map pose RMSE | Wrong fixes used |
| --- | --- | --- | --- | --- | ---: |
| person_tracking | none | 22.9% | 569 of 580 | 9.5 (9.4–9.6) cm | 0 |
| person_tracking | `--init-min-inliers 35 --min-inlier-cells 6` | 9.9% | **54** | 7.8 cm | 0 |
| moving_obstructing_box | none | 56.6% | 587 of 590 | 5.1 (5.0–5.3) cm | 1 |
| moving_obstructing_box | `--init-min-inliers 35 --min-inlier-cells 6` | 18.3% | **0** | — | 0 |
| removing_obstructing_box | none | 45.0% | 959 of 963 | 9.6 (9.1–10.1) cm | 6 |
| removing_obstructing_box | `--init-min-inliers 35 --min-inlier-cells 6` | 21.4% | **223** | 5.1 (4.7–5.1) cm | 0 |

Where it starts, the TUM setting is as accurate or better; but on these
sequences few fixes have 35 inliers, so it starts late or never. An
absolute inlier count does not transfer between maps and cameras; the start
needs a rule relative to what the map gives (for example the inliers of
the fixes seen so far, or agreement over a stretch of motion).

## Starting the correction only from fixes that agree across motion (negative)

`OdomCorrectionConfig::init_min_travel_m` / `init_min_turn_deg`
(`--init-min-travel-cm`, `--init-min-turn-deg`; off by default): the fixes
that set the first offset are drawn from the last 10 s and must agree *and*
span that much odometry travel or turn. The hypothesis was that walking_rpy's
biased first fixes, all from one view, would stop agreeing once the view
changed. **It does not hold**, and the rule has a fatal side effect.

Map pose RMSE, median of three seeds (range); frames with a map pose.

| Setting | TUM sitting_static | TUM walking_static | TUM walking_xyz | TUM walking_rpy | Bonn person_tracking | Bonn moving_obstructing_box | Bonn removing_obstructing_box |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1.5 cm, 701 | 2.2 cm, 737 | 2.0 cm, 855 | 10.0 cm, 901 | 9.5 cm, 569 | 5.1 cm, 587 | 9.6 cm, 959 |
| `--init-min-inliers 35 --min-inlier-cells 6` | 1.5 cm, 701 | 2.2 cm, 737 | 1.7 cm, 855 | 3.7 cm, 802 | 7.8 cm, 54 | none, 0 | 5.1 cm, 223 |
| 20 cm or 10° | 1.7 cm, 502 | **none, 0** | 2.5 cm, 823 | 9.8 cm, 885 | 8.3 (5.7–10.9) cm, 254 | 5.1 cm, 543 | 7.9 cm, 916 |
| 40 cm or 20° | **none, 0** | **none, 0** | 2.7 cm, 736 | 7.3 cm, 872 | 5.6 (4.8–8.5) cm, 250 | 7.4 cm, 506 | 9.5 cm, 825 |
| 20 cm or 10°, `--min-inlier-cells 6` | 1.7 cm, 502 | **none, 0** | 2.0 cm, 823 | 9.9 cm, 885 | 5.7 (4.9–6.4) cm, 254 | 5.1 cm, 543 | 7.9 cm, 916 |

- **walking_rpy is not fixed** (7.3–9.9 cm against 10.0): its camera turns
  quickly from the start, so the biased fixes span 10–20° and still agree.
  The bias is not tied to a single view.
- **A camera that does not move never gets a map pose** (walking_static,
  and sitting_static at 40 cm / 20°). For a robot that starts standing
  still this is disqualifying.
- On Bonn it trades frames with a map pose for accuracy, as the inlier
  threshold did, less sharply.

The switch stays in the code, off by default, with this result recorded;
it is not recommended. What starts walking_rpy right is the strength of the
first fixes (`--init-min-inliers`), which does not transfer to Bonn as an
absolute number. A threshold relative to the map's own fixes is the next
thing to try, not more agreement.

## Starting from the strongest fixes after a warm-up

`OdomCorrectionConfig::init_warmup_s` (`--init-warmup-s S`; off by
default): the correction waits S seconds, then seeds itself with the fix
that has the most inliers among the fixes of the last 10 s (the next
strongest if too few agree with it) and starts once `init_fixes` agree
with the seed. Relative to the fixes the map actually gives, not an
absolute count, and a still camera starts too. With it off, the outputs
are unchanged (checked on walking_xyz).

Map pose RMSE (max), median of three seeds; frames with a map pose. Bonn
counts fixes as wrong from 15 cm, and its ground truth carries about 5 cm
of calibration error.

| Setting | TUM sitting_static | TUM walking_static | TUM walking_xyz | TUM walking_halfsphere | TUM walking_rpy |
| --- | --- | --- | --- | --- | --- |
| none | 1.5 (2.8) cm, 701 | 2.2 (4.0) cm, 737 | 2.0 (5.9) cm, 855 | 3.1 (9.2) cm, 1062 | 10.0 (20.2) cm, 901 |
| `--init-min-inliers 35 --min-inlier-cells 6` | 1.5 (2.8) cm, 701 | 2.2 (4.1) cm, 737 | 1.7 (3.5) cm, 855 | 2.8 (8.0) cm, 1057 | 3.7 (8.0) cm, 802 |
| `--init-warmup-s 2` | 1.5 (2.3) cm, 642 | 1.9 (3.1) cm, 678 | 2.5 (11.0) cm, 796 | 2.9 (8.0) cm, 1003 | 3.8 (12.1) cm, 845 |
| `--init-warmup-s 3` | 1.4 (2.2) cm, 613 | 2.1 (3.3) cm, 648 | 2.0 (5.9) cm, 766 | 2.9 (7.7) cm, 973 | **2.9 (5.4) cm**, 812 |
| `--init-warmup-s 5` | 1.6 (2.4) cm, 553 | 2.0 (3.1) cm, 589 | 1.8 (5.9) cm, 706 | 2.8 (7.0) cm, 914 | 2.9 (5.4) cm, 687 |
| `--init-warmup-s 3 --min-inlier-cells 6` | 1.4 (2.2) cm, 613 | 2.1 (3.3) cm, 648 | 1.8 (3.8) cm, 766 | 2.8 (7.8) cm, 973 | 3.5 (6.3) cm, 812 |

| Setting | Bonn person_tracking | Bonn moving_obstructing_box | Bonn removing_obstructing_box |
| --- | --- | --- | --- |
| none | 9.5 (15.4) cm, 569 | 5.1 (9.1) cm, 587 | 9.6 (13.6) cm, 959 |
| `--init-min-inliers 35 --min-inlier-cells 6` | 7.8 (8.5) cm, 54 | none, 0 | 5.1 (5.7) cm, 223 |
| `--init-warmup-s 3` | **13.6** (17.1) cm, 273 | **11.8** (5.9–15.1) cm, 499 | 6.7 (10.6) cm, 871 |
| `--init-warmup-s 5` | 13.6 (17.1) cm, 273 | 7.6 (9.5) cm, 237 | 8.0 (10.6) cm, 806 |

- **On TUM it does what the inlier threshold did, without a number to
  tune:** walking_rpy 10.0 → 2.9 cm (worst 20 → 5.4 cm) with 3 s, the
  still-camera sequences start (3 s later), the others hold or improve.
- **On Bonn it does not carry over:** better on removing_obstructing_box,
  worse on the other two. There the fixes are few and weak (median 4–12
  inliers per image, most images not localized), and the strongest fix of
  the first seconds is not reliably right; Bonn's 5 cm ground-truth error
  also blurs the comparison.
- So no default changes. With a map that covers the place (TUM-like),
  `--init-warmup-s 3` is the start rule to use; with sparse, weak fixes
  (Bonn), none of the start rules measured here is reliably better than
  the default.

## Not yet measured

- **OpenLORIS-Scene cafe**, the Gazebo actors and masks: not started.
- **Detector cost on a Jetson:** needs hardware.

## Targets (proposed with Phase 0)

On the walking sequences, against the Phase 0 tables:

- single image: fixes more than 10 cm off down by at least half, losing at
  most a tenth of the good fixes;
- with an odometry: no fix more than 10 cm off used, walking_rpy's map pose
  under 5 cm RMSE, the control unchanged;
- under 1 ms added per localized image.

## Reproduce

```bash
D=/data/tum   # rgbd_dataset_freiburg3_{sitting,walking}_*
W=/work
cargo build --release --example basalt_euroc_localize_demo
cargo build --release --features image-io --example tum_rgbd_save_map
# Map: sitting_xyz + sitting_halfsphere, concatenated (rgb.txt, depth.txt and
# groundtruth.txt merged and sorted by stamp; rgb/ and depth/ linked).
target/release/examples/tum_rgbd_save_map --seq-dir $W/tum_combined/sitting_moving \
  --trajectory $W/tum_combined/sitting_moving/groundtruth.txt --out $W/maps/sitting_moving \
  --keep-world --keyframe-m 0.02 --keyframe-deg 2 \
  --fx 535.4 --fy 539.2 --cx 320.1 --cy 247.6 --k1 0 --k2 0 --p1 0 --p2 0 --k3 0
Q="--ratio none --max-hamming 60 --min-inliers 20 --threads 4"
for s in sitting_static walking_static walking_xyz walking_halfsphere walking_rpy; do
  python3 scripts/tum_rgbd_to_euroc.py --seq-dir $D/rgbd_dataset_freiburg3_$s --out $W/tum/$s
  # Single image
  target/release/examples/basalt_euroc_localize_demo --map $W/maps/sitting_moving \
    --euroc-dir $W/tum/$s --out-dir $W/q/sm_$s $Q --diagnostics $W/q/sm_$s/diagnostics.csv
  python3 scripts/eval_dynamic_localization.py \
    --gt $D/rgbd_dataset_freiburg3_$s/groundtruth.txt --label $s $W/q/sm_$s
  # With a drifting odometry, seeds 1-3
  for seed in 1 2 3; do
    python3 scripts/synth_drift_odometry.py --gt $D/rgbd_dataset_freiburg3_$s/groundtruth.txt \
      --out $W/odom/$s.seed$seed.tum --seed $seed
    target/release/examples/basalt_euroc_localize_demo --map $W/maps/sitting_moving \
      --euroc-dir $W/tum/$s --out-dir $W/q/od${seed}_$s $Q \
      --odometry $W/odom/$s.seed$seed.tum --localize-every 3 \
      --diagnostics $W/q/od${seed}_$s/diagnostics.csv
    python3 scripts/eval_dynamic_localization.py \
      --gt $D/rgbd_dataset_freiburg3_$s/groundtruth.txt --label "$s s$seed" $W/q/od${seed}_$s
  done
done
```

### Bonn

Data from `www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/` (intrinsics
from that page: fx 542.822841, fy 542.576870, cx 315.593520, cy 237.756098,
distortion 0.039903, −0.099343, −0.000730, −0.000144, 0).

| Archive | SHA-256 |
| --- | --- |
| `rgbd_bonn_static.zip` | `6920c6ef01593446915c82189eb0e0eb8e62a3fdb3b3d8ae46e1966fab9a5bb2` |
| `rgbd_bonn_static_close_far.zip` | `a9fa11127385728c167df64eae8dd36609f61c1cff4a3d9f2b9af79579677ce2` |
| `rgbd_bonn_crowd.zip` | `76b30c86e4ccb78ab668dd427dab696faef4885633dba436bc391b3a8142df70` |
| `rgbd_bonn_crowd2.zip` | `e751ca1b64165f3789d1c396d5c5c3d25e7ceb49cab01fad9ec87993f6244840` |
| `rgbd_bonn_person_tracking.zip` | `a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5` |
| `rgbd_bonn_moving_obstructing_box.zip` | `d0fc8c33d896b6a4c4a506508f44ddec63bf7dc37c8f58524826cbca1bbba689` |
| `rgbd_bonn_removing_obstructing_box.zip` | `1dfedd7516b4ce25a7598dcc67f27178377d6ef48353d597750890d671a2a132` |

```bash
B=/data/bonn   # rgbd_bonn_*
K="--fx 542.822841 --fy 542.576870 --cx 315.593520 --cy 237.756098"
# Camera in the marker frame, from the two static sequences (needs numpy, OpenCV)
python3 scripts/rgbd_hand_eye.py --seq-dir $B/rgbd_bonn_static_close_far $B/rgbd_bonn_static \
  $K --dist 0.039903,-0.099343,-0.000730,-0.000144,0 --gap 8 --offsets-ms=0:10:10 --max-pairs 200
X=-0.00933,0.10053,-0.02389,-0.125559,0.882537,-0.450645,0.047777
for s in static static_close_far crowd crowd2 person_tracking moving_obstructing_box \
    removing_obstructing_box; do
  python3 scripts/rgbd_hand_eye.py --seq-dir $B/rgbd_bonn_$s $K --apply=$X --out-trajectory $W/bonn/$s.cam.tum
done
# Map: static (first 337 s, where depth exists) + static_close_far, concatenated
# as for TUM, along the camera ground truth
target/release/examples/tum_rgbd_save_map --seq-dir $W/bonn_seq/static_both \
  --trajectory $W/bonn/static_both.cam.tum --out $W/maps/bonn_both --keep-world \
  --match-radius-px 8 --track-keyframes 10 --keyframe-m 0.05 --keyframe-deg 5 \
  $K --k1 0.039903 --k2 -0.099343 --p1 -0.000730 --p2 -0.000144 --k3 0
# Queries as for TUM, scored with --gt $W/bonn/$s.cam.tum --wrong-cm 15
```
