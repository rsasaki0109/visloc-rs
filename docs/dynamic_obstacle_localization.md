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

## OpenLORIS-Scene cafe: a real robot among people

The switches on a real wheeled robot's data, in a café with people about:
cafe1-1's map (stereo-inertial SLAM, `--smooth-features 1.5
--detector-quality 0.003`, as in
[openloris_map_localization.md](openloris_map_localization.md#two-more-scenes-cafe-and-home)),
cafe1-2 localized with the stereo-inertial VIO + map (`--vio
--localize-every 5`), with the demo's default matching settings (ratio test
on, 30 inliers), not the TUM ones. Two runs per setting gave identical
outputs. Fixes are scored with the local fit (the map around each, 5 m),
map poses with the map's fixed alignment, since many lie far from the
map's own path.

| Setting | Fixes (scored) | Fix error RMSE / median / max | Fixes within 5 cm | Map poses | Map pose RMSE / median / max (fixed) |
| --- | --- | --- | --- | --- | --- |
| none | 206 (166) | 7.7 / 5.9 / 18.3 cm | 43% | 2,698 | 18.1 / 15.5 / 33.9 cm |
| `--min-inlier-cells 6` | 195 (159) | 7.7 / 5.8 / 18.3 cm | 44% | 2,698 | 17.9 / 15.4 / 33.8 cm |
| `--init-warmup-s 3` | 263 (214) | 7.5 / 6.4 / 18.3 cm | 35% | 2,609 | 17.9 / 15.5 / 34.1 cm |
| both | 246 (204) | 7.5 / 6.4 / 18.3 cm | 36% | 2,609 | 17.6 / 15.2 / 33.8 cm |

- **Neutral here.** Neither switch changes the map pose by more than 0.5
  cm; the warm-up costs the first 3 s of map poses (91 frames). The
  café's people do not cause the failures the switches address: no fix is
  more than 18 cm off, and the start is not biased.
- **What limits the café is the map, not the people**: the map itself is
  17–19 cm from the ground truth (no loop closes; see the OpenLORIS page),
  and a fix comes every 10 s (median; 55 s at most), so the VIO carries
  the pose 2.9 m between fixes at the median.
- Per image 53–56 ms on 4 cores with either switch (the cell count is
  free; the warm-up only waits).

Data: `package/cafe1-1_2-package.tar` from the dataset's Hugging Face
mirror (revision `cbc03108723d08322b23d0338680bffa9404cce9`), SHA-256
`6f42810f87f21b3c720f517f0c514c6ba37ccf789c596f96e75df60226f43899`,
converted with `scripts/openloris_to_euroc.py`. CC BY-ND 4.0.

## RNE: the Dr Johnson interior with people walking through

[Robot Native Engine](https://github.com/rsasaki0109/RobotNativeEngine)'s
example 113 drives a diff-drive robot with a stereo camera (20 Hz, 0.35 m
up) and an IMU (60 Hz) round a loop in the photo-real Dr Johnson 3DGS
interior and exports EuRoC. With `RNE_WALKERS` (RNE branch
`claude/drjohnson-walkers`, commit `d9ac5cd`), it adds walkers as splat actors
that pace lines across the loop, sorted and rendered with the scene, so they
occlude it and are occluded correctly; their gait phases come from
`ros2/visloc-ros2/gs/walker_splat_plys.py`. Rendered on the CPU (Mesa
llvmpipe): about 10 minutes per 40 s episode.

- **Map:** an episode without people, through the stereo-inertial SLAM
  (`--save-map`): 220 keyframe images, 4,724 landmarks, 1.3 cm ATE.
- **Queries:** the same loop from another waypoint, with 0, 4 and 6
  walkers, and one with 4 walkers from the map's own start. VIO + map,
  `--localize-every 2` (10 Hz), the demo's default matching settings.
  Scored with the map's fixed alignment to the ground truth
  (`scripts/rne/rne_gt_to_euroc.py`, then `evaluate_map_localization.py
  --mode fixed`). The runs are deterministic.

| Query | Setting | Fixes, error RMSE / max | Map poses | Map pose RMSE / median / max |
| --- | --- | --- | --- | --- |
| no people | none | 392, 2.5 / **38.2** cm | 799 | 2.1 / 1.6 / 7.9 cm |
| no people | `--min-inlier-cells 6` | 389, 1.1 / 4.1 cm | 799 | 2.1 / 1.6 / 7.8 cm |
| no people | `--init-warmup-s 3` | 387, 3.0 / 54.3 cm | 739 | 2.1 / 1.6 / 8.7 cm |
| 4 walkers | none | 377, 1.1 / 7.5 cm | 799 | **4.7** / 1.7 / **20.4** cm |
| 4 walkers | `--min-inlier-cells 6` | 375, 1.1 / 7.5 cm | 799 | 4.7 / 1.7 / 20.4 cm |
| 4 walkers | `--init-warmup-s 3` | 383, 3.0 / 54.8 cm | 739 | **2.1** / 1.6 / **9.5** cm |
| 4 walkers | both | 380, 1.0 / 4.0 cm | 739 | 2.1 / 1.6 / 9.5 cm |
| 6 walkers | none | 380, 2.7 / **47.7** cm | 799 | 2.5 / 1.6 / 10.0 cm |
| 6 walkers | `--min-inlier-cells 6` | 378, 1.1 / 6.3 cm | 799 | 2.5 / 1.6 / 9.9 cm |
| 6 walkers | both | 381, 1.1 / 6.3 cm | 739 | 2.4 / 1.6 / 11.0 cm |
| 4 walkers, map's start | none | 379, 1.3 / 5.5 cm | 799 | 2.4 / 1.4 / 12.6 cm |
| 4 walkers, map's start | `--init-warmup-s 3` | 395, 1.5 / 5.5 cm | 739 | 2.3 / 1.3 / 12.9 cm |

- **The walkers cost little here.** They hide about 7% of the mapped image
  cells (93% matched, against 100% without them; TUM's walking people hid
  a third), and the map pose stays at 2.1–2.5 cm RMSE with 4 or 6 of them.
- **`--min-inlier-cells 6` removes the wrong fixes** (worst 38–48 cm to 4–6
  cm), with or without people, losing 2–3 fixes in 400. The correction's
  gate had already turned them away, so the map pose does not change.
- **`--init-warmup-s 3` fixes a bad start** on the 4-walker query (4.7 →
  2.1 cm, worst 20 → 9.5 cm). There the correction started from two good
  fixes at 0.07 s, while the just-started VIO was still settling; it then
  turned away good fixes (0.7 cm off) and the map pose was 8–20 cm off from
  0.4 to 4.5 s. Starting 3 s in avoids it. The warm-up's own column of fix
  errors (up to 55 cm) is the fixes it waits through, none of them used.
- Both switches cost nothing per image (52–56 ms either way).

**Next, to make it harder:** walkers that stop or turn in front of the
robot, and more of them, so that they hide a large share of the view as on
TUM; masks from the actors (ids or depth) for the share of inliers on them.

## RNE: people walking just ahead of the robot (the hard case)

RNE's `RNE_WALKER_MODE=ahead` (branch `claude/drjohnson-walkers`, commit
`3f57f7f`) keeps each walker 0.8–2.0 m in front of the robot, crossing
its view from side to side as it drives: people walking just ahead of a
robot. Two queries from waypoint 2, with 2 and 4 such walkers (`a2`, `a4`),
besides the earlier ones (`q0`: nobody, `q4`/`q6`: walkers pacing lines
across the route). With 4 of them, 69% of the mapped cells are matched (93%
with walkers across the route, 100% with nobody).

Map pose RMSE / median / max (fixed alignment), and the VIO's own ATE:

| Setting | q0 | q4 | q6 | a2 | a4 |
| --- | --- | --- | --- | --- | --- |
| none | 2.1 / 1.6 / 7.9 cm | 4.7 / 1.7 / 20.4 | 2.5 / 1.6 / 10.0 | **60.8** / 2.4 / 288 | **70.2** / 6.8 / 373 |
| `--min-inlier-cells 6` | 2.1 / 1.6 / 7.8 | 4.7 / 1.7 / 20.4 | 2.5 / 1.6 / 9.9 | 61.2 / 2.4 / 288 | 71.4 / 6.7 / 373 |
| `--init-warmup-s 3` | 2.1 / 1.6 / 8.7 | 2.1 / 1.6 / 9.5 | 2.4 / 1.6 / 11.0 | 22.8 / 2.3 / 207 | 12.2 / 6.4 / 71 |
| `--vio-map-prior` | 1.5 / 1.2 / 5.2 | 1.4 / 1.0 / 10.8 | 1.3 / 1.1 / 4.9 | 58.9 / 5.2 / 296 | 62.5 / 4.2 / 326 |
| **`--vio-map-prior --init-warmup-s 3`** | **1.2 / 1.1 / 3.7** | **1.2 / 1.0 / 8.8** | **1.2 / 1.0 / 5.4** | **20.4** / 2.8 / 207 | **7.1** / 2.8 / 31 |
| the same with `--min-inlier-cells 6` | 1.2 / 1.1 / 3.7 | 1.2 / 1.0 / 8.8 | 1.2 / 1.0 / 5.4 | 20.8 / 2.8 / 207 | 8.5 / 2.8 / 41 |
| VIO ATE, no setting | 3.6 cm | 4.6 | 3.4 | **68.2** | **46.6** |

- **The localization holds; the VIO does not.** With walkers ahead, the
  fixes stay accurate (worst 4.5 cm on `a4`), but the stereo-inertial VIO
  is dragged off: ATE 47–68 cm, up to 2.7 m, against 3–5 cm otherwise.
  People who keep pace with the robot look static to its camera and take a
  large share of the tracked points, so the VIO reads its own motion
  wrongly: failure F4 of the plan, not F1–F3.
- **The correction cannot follow a runaway VIO.** It is a yaw and a
  translation averaged over 5 s of fixes; when the VIO moves metres between
  fixes, its gate rejects even fixes 1 cm off and it resets over and over
  (`a2`, 3–5 s: map pose up to 2 m off with every fix within 2 cm).
- **`--vio-map-prior`, which puts each accepted fix into the VIO's window,
  together with the 3 s warm-up**, is the best setting on every query:
  1.2 cm on the three earlier ones (1.2–4.7 cm without), 7.1 cm on `a4`
  (70.2 without), 20.4 cm on `a2` (60.8). Neither alone does it: without
  the warm-up the map prior is fed from a correction started on the
  unsettled VIO; without the map prior the correction still chases the
  VIO. The cell gate adds nothing here.
- **What is left is the start of `a2`:** 2–5 s, before the map prior has
  pulled the VIO in, at up to 2 m.

### The VIO's moving-object filter with the walkers' true boxes (negative)

`basalt_euroc_localize_demo --vio --object-boxes FILE` (new) passes
camera-0 boxes to the VIO's moving-object filter
(`visloc_basalt::dynamic_objects`, Phase 2 of the plan).
`scripts/rne/rne_walker_boxes.py` writes the walkers' true boxes (each a
0.3 m × 1.8 m cylinder projected into cam0, scene occlusion ignored): the
filter's best case, before any detector.

| Query | Setting | Boxes judged moving / undecided / all | Map pose RMSE / max | VIO ATE |
| --- | --- | --- | --- | --- |
| a2 | none | — | 60.8 / 288 cm | 68.2 cm |
| a2 | boxes | 180 / 609 / 1,600 | 67.5 / 288 cm | 73.4 cm |
| a2 | `--vio-map-prior --init-warmup-s 3` | — | 20.4 / 207 cm | 52.2 cm |
| a2 | the same with boxes | 215 / 615 / 1,600 | 21.1 / 227 cm | 82.0 cm |
| a4 | none | — | 70.2 / 373 cm | 46.6 cm |
| a4 | boxes | 656 / 2,012 / 3,200 | **diverges** (hundreds of m) | diverges |
| a4 | `--vio-map-prior --init-warmup-s 3` with boxes | 588 / 2,009 / 3,200 | **diverges** | diverges |
| q4 | `--vio-map-prior --init-warmup-s 3` with boxes | 33 / 210 / 583 | 1.2 / 8.8 cm (unchanged) | 2.2 cm |

**It makes things worse.** The filter decides "moving" by comparing the
points in a box with the points outside every box. With walkers just ahead,
the boxes cover most of the image: too few background points remain for the
comparison (most boxes undecided), and where boxes are judged moving,
dropping their tracks leaves the VIO too few points, and with 4 walkers it
diverges. The filter was built for movers that take a small share of the
view (it was neutral on LaMAria for that reason); it is no answer when
people fill the view. `--object-boxes` stays, off by default.

What a robot has that these runs do not is an odometry that people cannot
fool: its wheels. Fusing the wheel odometry with the VIO and the fixes
(`--fusion --odometry`, `MapFusion`) is the next thing to measure here; RNE
knows the wheels' motion but example 113 does not export it yet.

## RNE: wheel odometry, which the walkers cannot fool

RNE's example 113 now also exports the base's wheel odometry
(`wheel_odom.tum`, branch `claude/drjohnson-walkers` commit `20b84e0`): the
true planar motion per step with a 2 % distance and 3 % turn error plus
noise, as the visloc-rs simulator's wheels. For the episodes rendered
before that, `scripts/rne/rne_wheel_odom_from_gt.py` computes the same from
their `gt.tum`: an `a2` episode rendered again with RNE `main` (`71bfc94`)
gives a byte-identical `gt.tum`, and its exported `wheel_odom.tum` is
byte-identical to the script's. The body is the IMU turned −90° about x
(`--imu-from-body 0,0,0,-0.7071068,0,0,0.7071068`).

Map pose RMSE (max), with the map fixes at 10 Hz:

| Odometry | q0 (nobody) | q4 | q6 | a2 (2 ahead) | a4 (4 ahead) |
| --- | --- | --- | --- | --- | --- |
| VIO (none of the switches) | 2.1 (7.9) cm | 4.7 (20.4) | 2.5 (10.0) | 60.8 (288) | 70.2 (373) |
| VIO, `--vio-map-prior --init-warmup-s 3` | 1.2 (3.7) | 1.2 (8.8) | 1.2 (5.4) | 20.4 (207) | 7.1 (31) |
| **wheels** (`--odometry`) | 2.4 (3.7) | 2.4 (3.7) | 2.4 (4.8) | **2.5 (4.2)** | **2.6 (4.1)** |
| VIO + wheels (`--vio --fusion --odometry`) | 1.5 (5.0) | 2.8 (14.3) | 2.0 (8.4) | 11.5 (90) | 11.8 (105) |
| VIO + wheels, `--vio-map-prior --init-warmup-s 3` | **1.1 (1.9)** | **1.0 (3.1)** | **1.1 (2.2)** | 5.2 (60) | 3.7 (24) |

- **The wheels + map hold everywhere**: 2.4–2.6 cm and never more than
  4.8 cm, with or without people in front. The walkers cannot move the
  wheels, and the fixes stay accurate.
- **Where the VIO is sound, adding it is better** (1.0–1.1 cm with the map
  prior and the warm-up, against 2.4 cm on the wheels alone).
- **Where it is not, it drags the fusion off** (a2/a4: 3.7–11.8 cm, up to
  1 m): `MapFusion` weighs the VIO by its fixed noise model, and nothing
  tells it that the VIO has run away while the wheels and the fixes
  agree.
- For a wheeled robot among people, then: **wheel odometry + map is the
  robust choice; VIO + wheels + map prior + warm-up the accurate one where
  the camera is not crowded.** The source check below tells the two apart
  online and gets both.

## Leaving a runaway odometry out of the fusion (source check)

`MapFusionConfig::source_check_window_s` (`--fusion-source-check-s S`; off
by default): at every fix, each odometry's motion since the oldest fix of
the last S seconds is compared with the motion between the two fixes. A
source that disagrees (more than 5 cm + 10 % of the distance, or 3°) has
its edges over that stretch left out of the graph, stays out of the nodes
added until a check finds it agreeing again, and is not used for the
current pose; the sources that agree carry it. If every source disagrees,
none is left out (the fixes are more likely wrong). It acts one fix
interval late: until the next fix nothing tells the sources apart.

**RNE** (map pose RMSE (max); `--vio --fusion --odometry wheel_odom.tum`,
fixes at 10 Hz, `--fusion-source-check-s 1`):

| Setting | q0 | q4 | q6 | a2 | a4 |
| --- | --- | --- | --- | --- | --- |
| wheels + map | 2.4 (3.7) cm | 2.4 (3.7) | 2.4 (4.8) | 2.5 (4.2) | 2.6 (4.1) |
| VIO + wheels, `--vio-map-prior --init-warmup-s 3` | 1.1 (1.9) | 1.0 (3.1) | 1.1 (2.2) | 5.2 (60) | 3.7 (24) |
| **the same with the source check** | **1.1 (1.9)** | **1.0 (3.1)** | **1.1 (2.2)** | **2.4 (37)** | **2.8 (22)** |
| VIO edges left out (of ~2,700) | 0 | 0 | 0 | 5 | 129 |

With the check, one setting is the best or level with the best on every
query: 1.0–1.1 cm where the camera is clear (the wheels alone: 2.4), and
2.4–2.8 cm with people just ahead of the robot (the wheels alone: 2.5–2.6).
The worst poses left (22–37 cm) are a few frames after the start (a2: 6
poses over 8 cm, all within 0.3 s; a4: 19).

**OpenLORIS cafe** (real robot, people about; VIO + wheels + map, fixed
alignment; the map's own 17–19 cm error dominates):

| Setting | Map pose RMSE (max) | Edges left out (VIO, wheels) |
| --- | --- | --- |
| VIO + map (above) | 18.1 (33.9) cm | — |
| VIO + wheels | 20.0 (35.5) | — |
| VIO + wheels, source check | **17.6** (34.3) | 6, 213 |
| VIO + wheels, `--vio-map-prior --init-warmup-s 3` | 21.0 (37.6) | — |
| the same with the source check | 20.7 (36.8) | 0, 6 |

No harm on real data. Here it is mostly the **wheels** that it leaves out:
the café run turns 915° in 90 s, and the wheels slip in turns (see the
OpenLORIS page). The check leaves out whichever source the fixes
contradict.

## On ROS 2

`visloc_map_localize_node` takes the two measured switches as parameters,
both off by default (behaviour unchanged):

- `min_inlier_cells` (e.g. 6): turn down a localization whose inliers
  occupy fewer cells of a 4 × 4 image grid.
- `init_warmup_s` (e.g. 3): before the first map pose, wait that long and
  start from the strongest fixes (most inliers) that agree. The node passes
  each fix's inlier count to `OdomCorrector::add_scored_fix`.

```bash
visloc_map_localize_node --map runs/map -p min_inlier_cells:=6 -p init_warmup_s:=3
```

They were measured offline on TUM and Bonn only (above): use them where
the map covers the robot's surroundings, and check them on the robot before
relying on them.

**Fusion and the source check** (`fusion:=true`, off by default): the node
subscribes `odom2` (e.g. the wheels; `odom` stays the VIO) and fuses both
with the fixes as the demo's `--fusion`. One difference: once the fusion
has a pose, it takes every full fix within 15 cm / 3 deg of **its own**
prediction, whatever the correction of `odom` made of it. The demo passes
only the fixes that correction accepted, which fails when `odom` runs away:
the correction then turns the good fixes away (a unit test: VIO running 3x
too fast after 6 s, wheels fine, 1.04 m off with the demo's rule, under
5 cm with the node's). In RNE, where the source check catches the VIO
within a fix interval, the node's rule costs a little offline:

| `--fusion-source-check-s 1`, VIO + wheels | q0 | q4 | q6 | a2 | a4 |
| --- | --- | --- | --- | --- | --- |
| + map prior + warm-up, demo's rule (above) | 1.1 (1.9) | 1.0 (3.1) | 1.1 (2.2) | 2.4 (37) | 2.8 (22) |
| the same, node's rule | 1.1 (1.9) | 1.0 (3.1) | 1.1 (2.2) | 3.2 (37) | 3.4 (23) |
| no map prior or warm-up, demo's rule | 1.4 (4.3) | 2.7 (14) | 1.7 (6.5) | 11.2 (90) | 11.0 (105) |
| the same, node's rule | 1.5 (5.3) | 2.8 (14) | 1.7 (6.4) | 11.1 (90) | 12.3 (105) |

The demo keeps its rule (the numbers above stay as they are).

**Through the nodes over DDS** (`euroc_dds_replay --rate 0.2 --odometry
wheel_odom.tum` into `visloc_vio_node` and `visloc_map_localize_node
localize_every:=2`, one 4-core machine; `map_pose` scored as above; each
run once):

| Setting | q0 | a4 |
| --- | --- | --- |
| default (VIO only) | 2.2 (7.0) cm | 71.6 (353) |
| `fusion:=true fusion_source_check_s:=1` | 1.8 (6.4) | 11.2 (92) |
| ... + `init_warmup_s:=3`, VIO `map_prior:=true` | **1.1 (2.4)** | **3.6 (31)** |

```bash
visloc_map_localize_node --map runs/map -p fusion:=true -r odom2:=wheel/odom \
  -p odom2_body_offset:=0,0,0,-0.7071068,0,0,0.7071068 \
  -p fusion_source_check_s:=1 -p init_warmup_s:=3 -p fusion_seed_s:=3
```

The nodes track the offline runs (default a4: 70.2 cm offline; last row:
3.4 / 1.1 offline with the node's rule).

## Starting the fusion from the warm-up's fixes (`--fusion-seed-s`)

The worst poses left above (a2: 37 cm, a4: 22 cm) come in the first two
seconds of the map pose. The wheels alone have none there (a2, a4: 1.7 /
1.4 cm at most in the first second, `--init-warmup-s 3`): it is the VIO,
not yet right after people blocked the start, that the fusion follows. The
source check cannot tell yet: the fusion starts from the one fix the
correction starts from, and the check needs a second of fixes.

`--fusion-seed-s S` (demo) / `fusion_seed_s` (node), off by default: when
the correction initializes or resets, the fusion starts from the full fixes
of the last S seconds that agree with that fix under one odometry's motion
(10 cm + 10 % of the distance, 3 deg;
`visloc_localization::fixes_agree_under_a_source`), oldest first. The
source check then has their history at once.

RNE, offline (map pose RMSE (max); first second of the map pose, max):

| `--vio-map-prior --init-warmup-s 3 --fusion-source-check-s 1` | q0 | q4 | q6 | a2 | a4 |
| --- | --- | --- | --- | --- | --- |
| as above | 1.1 (1.9) | 1.0 (3.1) | 1.1 (2.2) | 2.4 (37); 37 | 2.8 (22); 3.6 |
| `--fusion-seed-s 3` | 1.1 (1.8) | 1.1 (3.1) | 1.1 (2.3) | **1.9 (4.5); 1.3** | **2.2 (14); 1.4** |

The start-up errors are gone; a4's worst pose left (14 cm) is mid-run.
Through the nodes over DDS (one core, real time, one run each), a4 goes
from 2.0 (12) to **1.9 (10)** cm, its first second from 9.9 to 1.2 cm at
most; a2 (1.2 (4.1)) and q0 (1.1 (2.0)) are unchanged. On OpenLORIS cafe
it does no harm: 17.6 -> 17.7 cm (`--fusion-seed-s 1`, source check only)
and 20.7 -> 20.4 cm (with the map prior and warm-up, `--fusion-seed-s 3`).
A unit test (VIO off from the start, 3 s warm-up): 40 cm -> 0.

## A real person detector instead of the true boxes (YOLOX)

The boxes above were RNE's true walker boxes. With a detector
(`yolox_track_demo --boxes-csv`, the official YOLOX ONNX models of
`docs/object_tracking.md`, ONNX Runtime 1.32 on the CPU) on cam0:

| Model | Recall of the true boxes (a2, a4) | False detections | Time per frame, one Xeon core |
| --- | --- | --- | --- |
| `yolox_nano` (416) | 79 %, 81 % | 0 | 12 ms |
| `yolox_s` (640) | 84 %, 88 % | 0 | 64 ms |

(`scripts/rne/rne_box_recall.py`: true boxes of at least 2,000 px, found
at IoU >= 0.3. The true boxes ignore walls, so a walker hidden behind one
counts as missed: the recall is a lower bound. The time includes reading
the PNG and the tracker.)

The boxes into the VIO's moving-object filter, on top of the best setting
(`--vio-map-prior --init-warmup-s 3 --fusion-source-check-s 1
--fusion-seed-s 3`; map pose RMSE (max), VIO ATE):

| Boxes | a2 | a4 |
| --- | --- | --- |
| none | 1.9 (4.5) cm, VIO 51 cm | 2.2 (14) cm, VIO 36 cm |
| true | 1.7 (4.5), VIO 81 | **4.0** (14), VIO **242** |
| `yolox_nano` | 1.9 (5.0), VIO 51 | 2.0 (14), VIO 36 |

**Not worth its cost here.** The fusion already leaves the dragged VIO out,
so the filter has little left to fix: the detector's boxes change the map
pose by 0.2 cm at most, and the true boxes (more of them, people hidden
behind walls included) still break the VIO on a4, as above. A detector at
the camera rate would cost about a quarter of a Xeon core (`yolox_nano`)
for nothing measurable; on a Jetson its GPU could run it, but the budget is
better left to the VIO. The detector stays out of the recommended setting.

## A map built with people walking about (F3)

RNE example 113 rendered the mapping run again with six walkers wandering
the scene (`RNE_WALKERS`, `RNE_WALKER_COUNT=6`, the same path as the clean
map), and `basalt_euroc_online_slam_demo --save-map` built a map from it:
5,362 landmarks over 296 keyframes (the clean map: 4,724 over 220), SLAM
ATE 1.8 cm (clean: 1.3). The queries were localized against each map
(map pose RMSE (max); fixes: RMSE (max)):

| Query | Setting | Clean map | Map with walkers |
| --- | --- | --- | --- |
| q0 | default (`--vio`) | 2.1 (7.9) cm; fixes 2.5 (38) | 2.1 (6.9); fixes 1.3 (6.5) |
| q4 | default | 4.7 (20) | 3.2 (21) |
| a4 | default | 70.2 (373) | 80.4 (373) |
| q0 | best (fusion, source check, warm-up, map prior, seed 3) | 1.1 (1.8) | 1.0 (4.0) |
| q4 | best | 1.1 (3.1) | 1.1 (4.8) |
| a4 | best | 2.2 (14) | 2.5 (13) |

**No harm from people who keep walking.** Their points do not hold still
across the keyframes, so they are not triangulated into the map: counting
the landmarks within 0.45 m of a walker at an image that observes them
finds 9.9 % in the map with walkers, but 15 % in the clean map against the
same walker positions. The count catches background points near a
walker's path, not points on walkers. The worst fixes (38–55 cm, against
either map) are turned away by the correction or the fusion.

Not measured: people who stand still while the map is built (a queue, a
person at a desk) and are gone at localization. Their points would be
triangulated like any static surface; RNE's walkers do not stop.

## Compute budget (towards a Jetson)

No Jetson here, so the budget was measured on this machine's Xeon
(2.3 GHz) by pinning `visloc_vio_node` and `visloc_map_localize_node`
together to fewer cores (`taskset`; the player on its own core), replaying
in **real time** (20 Hz stereo + 200 Hz IMU, 40 s episodes) and faster (2x,
3x: as on a CPU that much slower). Setting: the last row above (fusion,
source check, warm-up, map prior), `localize_every:=2`. One run each; a
repeat of one case differed by a run with dropped frames, so read single
rows with care.

| Cores, speed | VIO config | q0 | a4 | CPU (both nodes) | VIO frames dropped |
| --- | --- | --- | --- | --- | --- |
| 1 core, 1x | default | 1.1 (2.3) cm | 2.0 (12) | 0.6-0.75 core | 0-5 of 783 |
| 1 core, 2x | default | 1.7 (11) | 3.9 (27) | | 17-241 |
| 1 core, 3x | default | — | 11.6 (69) | | 197 |
| 1 core, 1x | light | — | 4.4 (75) | 0.5 core | 0 |
| 1 core, 2x | light | 1.1 (3.0) | 2.5 (42) | | 20-28 |
| 1 core, 3x | light | — | 8.5 (57) | | 99 |

- **One Xeon core keeps up in real time**, with about 0.6-0.75 of it
  busy (VIO about 70 %, map localization the rest) and 210 MB of memory
  (VIO 120 MB, map localization 90 MB with this 4,700-landmark map). 2-3
  cores (1x) gave the same numbers.
- **At 2x** (a CPU core half as fast; a Jetson Orin's Cortex-A78AE core at
  1.5-2.2 GHz is roughly there, an estimate, not measured) the default
  config drops up to a third of the VIO frames on q0 (busier images) but
  the map pose holds. The **light** VIO config
  (`optical_flow_detection_grid_size` 50 -> 70, about half the points;
  `vio_max_kfs` 7 -> 5; `configs/basalt/variants/official_euroc_ds/euroc_config_light.json`) holds both at 2x with few drops. **At 3x** both
  degrade.
- Not levers: `max_iterations:=2` (the VIO's time is not in its LM
  iterations) and `localize_every:=4` (no better at 2x).
- So: **two Orin cores for the two nodes** should hold real time with the
  default config, one core with the light one. A Jetson run is still
  needed to confirm it.

**A fault the budget runs found.** With the light config at 1x, the VIO
broke down on a4 (people ahead) and the node's map pose ended up 7.6 m off.
The correction of the VIO rejected every fix after it (322). The fusion,
gating fixes only on its own prediction, never took one again once it was
more than 15 cm off, though the wheels were fine. The fusion now restarts
from `reset_after` (3) consecutive fixes it rejected that move as one
odometry did between them, as the correction resets. Same run: 757.7 cm
-> **4.4 cm**; the unit test (no fixes for 2 s while the VIO runs off):
1.04 m -> under 5 cm. Not repeated with the default config: a4 1x went
from 3.6 to 2.0 cm, q0 1.1 both.

## Not yet measured

- The Gazebo actors and masks: not started.
- **Detector cost on a Jetson:** needs hardware (on a Xeon core: above).
- **The nodes on a Jetson:** estimated above from a slowed Xeon only.
- **A map with people standing still while it was built** (F3's harder case): RNE's walkers do not stop.

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
