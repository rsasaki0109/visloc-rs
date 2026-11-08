# LaMAria: the recordings refined together

Each recording's VIO and BA chain runs on its own ([lamaria_submission.md](lamaria_submission.md)). The
recordings, though, cover the same few streets again and again. Here they are refined together, offline:
- Where two recordings pass the same place, a keyframe of one is localized in the other's map.
- A pose graph over every recording's keyframes then takes these cross edges with each recording's own odometry.
- GPS proposes which keyframes to try and places each recording roughly before the solve.

**Withdrawn (2026-11-08).** No submission uses this any more. Each sequence's estimate again comes from its own
recording only, and the leaderboard entry goes back to submission 7. The scripts stay as an experiment.
- **The rules are silent.** The challenge's rules
  ([documentation](https://lamaria.ethz.ch/slam_documentation)) say nothing about using other recordings,
  offline processing or sensors.
- **The other methods appear to solve each sequence alone.** From Short to Long, Aria's SLAM loses 19.8 points,
  SLAM Dunk 18.5 and GroundedSLAM 16.2: the drift of a longer walk. Submission 9 loses 0.3, because other
  recordings hold its long walks. Their descriptions do not say either way.
- **So a score from the whole test set together is not comparable** with theirs.
- **Its results guide nothing.** Submission 9's test scores and its estimates are not used to choose, tune,
  initialize or supervise the single-session methods.

The six training walks were solved inside the test graph, and the settings were chosen on their scores; no test
ground truth was used.

Submission 8 (2026-11-07) refined 49 test walks this way, but it was never evaluated: the site was unreachable
until submission 9 replaced it. Submission 9 adds three walks with poor GPS and eight of the ten Moving recordings
([Adding recordings to the solved graph](#adding-recordings-to-the-solved-graph)). On the leaderboard
(2026-11-08) every track gains 18–33 points, and the mean goes from 57.7 to 81.3:

| | Short | Medium | Long | Low light | Moving | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Submission 7 | 69.7 | 56.8 | 54.1 | 60.6 | 47.3 | 57.7 |
| Submission 9 (withdrawn) | 87.8 | 81.6 | 87.5 | 82.7 | 66.7 | 81.3 |

[lamaria_test_results.md](lamaria_test_results.md) has the sequences. On the six training walks inside the test
graph, the chain scores 445.3 in total and submission 8's setting 546.8:

| Sequence | The chain | The first build | Submission 8 |
|---|---:|---:|---:|
| sequence_1_19 | 82.25 | 89.09 | 91.33 |
| sequence_1_20 | 88.65 | 90.78 | 93.33 |
| sequence_2_11 | 89.25 | 95.23 | 95.02 |
| sequence_3_18 | 54.27 | 81.98 | **91.14** |
| sequence_4_10 | 59.80 | 60.77 | **87.89** |
| sequence_4_11 | 71.07 | 72.55 | **88.13** |
| Total | 445.29 | 490.40 | **546.84** |

The first build kept the settings that suit the training recordings alone. Submission 8 replaced them with the
settings chosen inside the test graph ([Choosing the settings](#choosing-the-settings)).

## The recordings overlap

All 68 recordings with GPS lie within 1.3 × 1.2 km. For each recording, the median share of its path that lies
within 20 m of another recording is 100 %. In the test graph, two thirds of the query keyframes (23,838 of
36,382) have three or more other recordings within 12 m.

## The method

The scripts run in this order:

1. **`scripts/lamaria_xs_keyframes.py`** (test recordings) fetches only the keyframes.
   - The 63 test zips total 947 GB. This takes the 1 Hz keyframes' PNGs from the remote zip by multi-range HTTP
     requests: about 0.5 GB and 4–5 minutes a recording at about 3 MB/s.
   - It extracts ALIKED features (1,024 keypoints; the chain's denoising and CLAHE on dark night frames).
   - It writes the submitted estimate as the recording's fixed trajectory, with 2 Hz pose graph keyframes.
   - For the training recordings, every other keyframe of the chain's own ALIKED features is used instead.
2. **`scripts/lamaria_xs_map.py`** builds a recording's own map from its fixed trajectory.
   - ALIKED + LightGlue between the two cameras, and to the same camera 1, 2 and 4 keyframes on.
   - Midpoint triangulation, keeping points within 2 px, 0.3–60 m in front and with 1° of parallax.
3. **`scripts/lamaria_xs_match.py`** makes the cross edges.
   - Each trajectory is placed by a rough 2D fit to its fixes.
   - Every 4th 1 Hz keyframe of a recording tries its two nearest other recordings within 12 m.
   - Against each map, the query's ALIKED keypoints (each camera) are matched by LightGlue to the landmark
     keypoints of the 3 nearest map keyframes within 60° of heading, then PnP RANSAC with 2.5 px.
   - The edge is the query's IMU pose relative to the nearest map keyframe, in that keyframe's heading frame, with
     the inlier count.
   - The maps load one at a time, so 55 recordings fit in 11 GB. The script is resumable.
4. **`scripts/lamaria_xs_graph.py`** solves the pose graph.
   - The nodes are the 2 Hz keyframes, with x, y, z and the heading of the IMU's most horizontal axis, y on Aria.
     The world-z components of the x, y and z axes average 0.68, 0.27 and 0.42, so a heading from x would be
     ill-defined. Roll and pitch stay each recording's own.
   - Odometry edges join consecutive keyframes, with sigma 5 mm + 4 % of the step and 0.005° a step.
   - Cross edges with 30 or more inliers get 0.15 m and 0.25°.
   - First each recording is placed as a rigid whole, then Gauss-Newton runs with Cauchy weights on the cross
     edges.
   - Then comes one round of per-recording odometry weights (variance components): each recording's odometry
     sigmas scale by how far it had to bend, relative to the median recording, and the graph is solved again.
   - Each trajectory is corrected between its keyframes by the interpolated change and written as TUM.
5. Then the usual steps: `basalt_tum_to_lamaria_estimate.py`, a check against the recordings' GPS
   (`lamaria_gps_agreement.py`), and the zip (`lamaria_assemble_submission.py`, which refuses an estimate whose
   timestamps differ or that holds a NaN).

The odometry weights loosen the recordings that drift. In submission 8's solve the scales run from 0.44 to 2.20
(median 1.00). The loosest are sequence_2_2, 3_3, 2_5, 1_18, 1_2 and 2_7. sequence_3_3 and 1_2 were among the
weakest test walks ([Notes for the next round](lamaria_test_results.md#notes-for-the-next-round)). Among the
training walks, night 4_10 gets 1.45 and day 1_19 gets 0.66.

## The training recordings alone

The first test used the eight training recordings 1_19, 1_20, 2_11, 3_18, 4_10, 4_11, 5_11 and 5_12, with 100+
inliers, odometry 1 % and cross edges 0.3 m and 0.5°.

| Queries (1 Hz images) | Total | Against the chain |
|---|---:|---:|
| Every 2nd | 510.1 | +7.6 (as at 2 Hz) |
| Every 4th | 509.6 | +7.1 |
| Every 6th | 508.6 | +6.0 |

By sequence, the gains are 1_19 +2.8, 1_20 +1.2, 2_11 −0.6 and 3_18 +4.5. The night and Moving recordings stay
about the same: each has few neighbours among the eight.

- **The edges are accurate.** Against the pseudo ground truth, the horizontal error has a median of 0.12 m. No
  edge with 100+ inliers is off by more than 2 m.
- **SIFT across recordings mostly fails,** because the viewpoint and the sun differ.
- **On this sparse graph, the test graph's settings do worse:** 499.5 against 510.1. The looser odometry needs
  neighbours along the whole recording to hold it.

## The test graph

The test graph has 55 recordings: 49 test walks and the six training walks above, without the two Moving ones.

Submission 8 leaves these walks out, and they keep their chain estimates. Submission 9 adds the first three
([Walks with poor GPS](#walks-with-poor-gps)).
- sequence_1_16 has 12 fixes.
- sequence_1_17, 2_10 and 4_7 have no fix with an accuracy of 10 m or better.

Three matching runs make the edges:

| Run | Queries | Edges | Time |
|---|---:|---:|---:|
| Every 4th 1 Hz keyframe against its 2 nearest maps | 36,382 | 14,910 | 1.8 h |
| The 1 Hz keyframes between those (`--offset 2`) | 36,313 | 14,899 | 3.2 h * |
| The first run's queries against their 3rd and 4th nearest maps (`--map-from 2 --max-maps 4`) | 23,838 | 8,982 | 3.0 h * |

\* The GPU was shared with other jobs. With three jobs on the 8 GB GPU, one of them stopped on
`CUDA_ERROR_UNKNOWN`; it resumed from its last map. One solve of the whole graph takes 6–8 minutes.

### Choosing the settings

All settings were scored on the six training walks inside the test graph. Each row adds one change to the row
above.

| Setting | Total |
|---|---:|
| The chain alone | 445.3 |
| The training settings: 100+ inliers, odometry 1 %, cross edges 0.3 m and 0.5° (the first build) | 490.4 |
| 30+ inliers | 505.6 |
| Odometry 2 %, cross edges 0.15 m and 0.25° | 518.8 |
| Odometry 4 % | 526.5 |
| One round of per-recording odometry weights (`--vce 1`) | 537.1 |
| The edges of all three runs (38,791) | **546.8** |

The other settings tried, each against its neighbour in the table:

- **Inlier floor** (training settings): 50+ gives 500.9, 150+ 485.7 and 200+ 478.7.
- **Odometry translation:** 0.5 % gives 485.3 and 2 % 497.7 (training settings); 8 % gives 526.1 against 4 %'s
  526.5.
- **Odometry heading:** 0.0025° gives 489.4 and 0.01° 487.2 (training settings); 0.01° at 4 % gives 524.9.
- **Cross edges:** 0.6 m and 1° give 482.0 (training settings); 0.1 m and 0.15° give 520.8 at 2 %.
- **Per-recording weights:**

  | Rounds | 0 | 1 | 2 | 3 | 4 |
  |---|---:|---:|---:|---:|---:|
  | Total | 526.5 | **537.1** | 534.4 | 529.3 | 528.0 |

  Two rounds at 2 % and 8 % give 529.6 and 529.7.
- **Edges:** with one round of weights, the first run's edges plus the second run's give 541.3, and plus the
  third run's 546.3.
- **Solver:** 40 Gauss-Newton iterations give the same as 15. A Cauchy scale of 2 sigmas gives 522.2 against 3's
  526.5.

About 35 settings were tried on six sequences, so the final total is the best of many and a little optimistic. The
steps kept are each several points, though, and they lift most of the six. One round of per-recording weights
raises all six (+0.0 … +3.8). The extra edges raise the night walks most (4_10 81.0 → 87.9, 4_11 85.9 → 88.1)
and cost 2_11 0.9.

### Agreement with GPS

The test estimates were checked against the recordings' own GPS, without ground truth
(`lamaria_gps_agreement.py`). Each estimate is fitted to its fixes by a robust 2D similarity, and the residuals
are compared. The fixes are off by about 3 m in the streets, so the check sees metre-level changes only.

| Change | Median of the medians | Better in | Worse by 0.5 m or more in |
|---|---:|---:|---:|
| The chain → the first build | 3.33 → 3.04 m | 38 of 49 | 0 |
| The first build → submission 8 | 3.04 → 3.03 m | 31 of 49 | 0 |

The walks submission 8 moved most also agree better with GPS:
- sequence_2_2 moved 3.2 m (median after a rigid fit), and its residual fell from 3.78 to 3.56 m.
- sequence_2_3: 3.36 → 3.14 m.
- sequence_4_5: 3.69 → 2.81 m.
- sequence_4_3: 2.85 → 2.63 m.
- sequence_4_8: 4.85 → 4.44 m.

## Adding recordings to the solved graph

Submission 9 adds recordings to submission 8's graph and leaves the 55 already solved as they are.

### Held recordings

`lamaria_xs_graph.py --hold DIR` takes the recordings with a trajectory in `DIR`, an earlier solve's output, as
solved:
- They stay where that solve put them, through the rigid placement and Gauss-Newton.
- The edges among them count for nothing.
- Only the other recordings are solved and written.

Solving everything again would move the 55 as well. With the three walks below joined in one joint solve, the six
training walks fall from 546.8 to 544.6 (4_10 87.9 → 84.7). Held instead, the 55 stay as submission 8 byte for
byte. The three walks come out within 0.13–0.20 m of the joint solve (median, after a rigid fit).

### Walks with poor GPS

The matcher places each recording by its fixes with an accuracy of 10 m or better, and leaves out a recording
with fewer than 30 of them. Such walks join in two matching runs, with `--loose` and `--only` naming them:

1. **Stage 1.** They take fixes up to 50 m (10 are enough). Every 10th 1 Hz keyframe tries the 4 nearest maps
   within 30 m, against 5 map keyframes (`--radius 30 --window 5 --every 10 --max-maps 4`).
2. **Stage 2.** Each is placed by the first run's edges with 100+ inliers instead (`--place`; a robust 2D fit,
   Cauchy 3 m). Every 2nd keyframe then tries the 4 nearest maps within 12 m (`--every 2 --max-maps 4`).

In the graph these walks start from fixes up to 50 m.

**A training check.** Two training walks, 1_20 and 4_11, were given poor GPS: a third of their fixes, each off by
a slowly wandering 20 m error (16.1 and 22.6 m at the median), with an accuracy of 40 m. Their 1,466 edges were
taken out of submission 8's 38,791.
- Stage 1 found 267 edges.
- Stage 2 placed the two walks 7.3 and 9.3 m from where their GPS had put them, and found 1,845 edges.

Solved in the test graph, the six training walks score:

| Sequence | Left out (the chain) | Joined in two stages |
|---|---:|---:|
| sequence_1_20 | 88.65 | 93.64 |
| sequence_4_11 | 71.02 | 90.06 |
| The other four | 365.21 | 367.26 |
| Total | 524.88 | **550.96** |

**On the test.** sequence_1_16, 1_17, 2_10 and 4_7 were left out of submission 8.
- Stage 1: 1,778 queries, 514 edges, 14 minutes.
- Stage 2 placed 1_16 by 45 edges, 19.5 m from where its GPS had put it. It placed 1_17 by 127 edges (7.7 m) and
  2_10 by 114 (9.4 m). It then found 3,298 edges from 6,737 queries in 34 minutes.
- In the held solve the three walks have 931, 1,441 and 1,552 cross edges.
- sequence_4_7 gets no edge: its GPS puts it about 1 km from every other recording. It keeps its chain estimate.

On the leaderboard, 1_16 goes from 35.7 to 76.1, 1_17 from 53.9 to 90.4, and 2_10 from 73.0 to 88.0.

### The Moving recordings

The Moving recordings walk and ride, a tram for instance. Submission 7 bridges their missed rides by GPS
([lamaria_gps.md](lamaria_gps.md#test-set-submission-7)).

**Training.**
- **The two training Moving recordings,** joined to submission 8's graph, do not gain together:
  - sequence_5_11 falls from 51.46 to 40.93. Its ride, missed before its first good fix, leaves its first CP 35 m
    off, and the edges then bend the rest of it.
  - sequence_5_12 goes from 5.80 to 7.62.
- **Bridged rides gain.** The GPS bridges made on purpose ([Missed rides on the training
  sequences](lamaria_gps.md#missed-rides-on-the-training-sequences)) were put in the graph in place of five
  training walks. The graph takes back 17–29 points of each ride:

  | Sequence | Without the ride, in the graph | 90 s missed, bridged by GPS | The same in the graph |
  |---|---:|---:|---:|
  | sequence_1_20 | 93.33 | 47.79 | 76.77 |
  | sequence_2_11 | 95.02 | 57.26 | 79.45 |
  | sequence_3_18 | 91.14 | 43.78 | 68.85 |
  | sequence_4_10 | 87.89 | 57.32 | 80.80 |
  | sequence_4_11 | 88.13 | 68.67 | 85.48 |

- **Looser odometry on the ride** (`--fast-speed 3`, with 50 % or 20 % of the step) changes the six training
  walks' total by 0.02. It is not used.

The test Moving recordings start from submission 7's estimates, bridged except 5_8 and 5_10, where the bridging
found no jump.

**On the test.**
- **The keyframes.** The ten recordings' keyframes came to 9.0 GB, fetched and mapped in about an hour.
- **The edges.** One matching run with `--only` covered the pairs that include a Moving recording, among all 65
  recordings: every 2nd 1 Hz keyframe against the 4 nearest maps. It found 15,593 edges from 50,103 queries in
  2.6 hours.
- **The solve.** The ten recordings were then solved with the 58 walks held.

| Sequence | Cross edges | Odometry scale | GPS residual, median | GPS residual, 90 % | Score, submission 7 → 9 |
|---|---:|---:|---:|---:|---:|
| sequence_5_1 | 1,695 | 1.04 | 3.88 → 3.05 m | 7.83 → 7.88 m | 43.4 → 68.2 |
| sequence_5_2 | 3,380 | 0.90 | 3.72 → 3.44 m | 9.07 → 7.85 m | 51.7 → 82.8 |
| sequence_5_3 | 2,826 | 1.00 | 3.90 → 3.00 m | 8.12 → 7.58 m | 31.0 → 41.9 |
| sequence_5_4 | 2,037 | 1.16 | 4.51 → 3.56 m | 15.31 → 9.97 m | 36.2 → 69.0 |
| sequence_5_5 | 2,092 | 0.70 | 4.12 → 4.23 m | 9.99 → 10.21 m | 63.7 (kept) |
| sequence_5_6 | 1,114 | 1.00 | 4.32 → 3.67 m | 12.50 → 10.41 m | 35.2 → 62.4 |
| sequence_5_7 | 1,512 | 0.75 | 3.03 → 3.39 m | 6.73 → 6.85 m | 69.2 (kept) |
| sequence_5_8 | 1,678 | 0.57 | 2.97 → 2.81 m | 7.21 → 7.27 m | 72.3 → 88.3 |
| sequence_5_9 | 835 | 1.49 | 3.60 → 2.66 m | 12.21 → 5.97 m | 37.6 → 62.5 |
| sequence_5_10 | 1,806 | 0.94 | 2.79 → 2.75 m | 6.16 → 6.17 m | 32.8 → 59.4 |

- **The rule was set before the solve.** A recording whose median GPS residual grew would keep submission 7's
  estimate. That kept sequence_5_5 (+0.11 m) and 5_7 (+0.36 m).
- **The eight joined all gain,** by 10.9 to 32.8 points. The Moving track goes from 47.3 to 66.7.
- **The GPS residual hardly foretold the gains.** sequence_5_10's median moved by 0.04 m and its score by 26.6.

### GPS in the solve: a negative result

`--gps-sigma` puts every fix into the solve, with Cauchy weights at 2 sigmas. It does worse on the dense graph:

| Graph | Without | 30 m | 10 m | 3 m |
|---|---:|---:|---:|---:|
| The test graph (the six training walks) | **546.8** | 544.0 | 525.2 | 465.9 |
| The eight training recordings (training settings) | 510.1 | **511.7** | 510.8 | 500.3 |

The fixes are off by about 3 m in the streets, while the cross edges miss by 0.05 m. The GPS terms pull the
recordings apart where the edges already hold them. On the sparse graph the gain at 30 m (+1.6) is within the
noise of the settings.

## Other matchers: a negative result

FlexGlue (2026) and the matchers of VisMatch were tried in place of ALIKED + LightGlue.

**Fixed pairs.** The test drew 637 GPS-proposed pairs from the eight training recordings:
- Each query keyframe was paired with its nearest map keyframe, up to 250 pairs per kind, matched left camera to
  left camera.
- A pair counts as localized with 30+ PnP inliers.
- The error is the query's horizontal error, with the map trajectory fitted locally to the pseudo ground truth.

| Matcher | Day–day (250) | Day–night (250) | Night–night (137) |
|---|---|---|---|
| ALIKED + LightGlue (the stored features) | 60 localized, 0.09 m | 5, 0.39 m | 7, 0.09 m |
| FlexGlue (the map's landmark keypoints predicted in the query) | 128, 0.13 m | 26, 0.20 m | 26, 0.14 m |

FlexGlue localizes 2–5 times as many pairs. Its errors are larger: 6 % and 9 % of its day–day and night–night
edges are off by more than 2 m. On its own on the 8 GB GPU, it takes 43 ms to describe a frame and 142 ms a pair.

**End to end, on the eight training recordings.** FlexGlue's run used every 2nd 1 Hz keyframe, both query
cameras, and 3 map keyframes × 2 cameras.

| Cross edges | Edges | Training settings | Test graph's settings |
|---|---:|---:|---:|
| ALIKED + LightGlue | 721 | **510.1** | 499.5 |
| FlexGlue | 1,309 | 504.1–505.4 (floors 30–200) | 493.4 |
| Both | 2,030 | | 493.3 |

More edges do not lift the graph. FlexGlue's edges are noisier: with 100+ inliers, the solved graph still misses
them by 0.11 m at the median, against 0.08 m for LightGlue's.

**The cost.** The run took 6,237 s against LightGlue's 914 s, with the GPU shared. The test set would also need
its keyframe images again (about 25 GB), since only the ALIKED features were kept.

**VisMatch's dense matchers.** RoMa v2 at 8,000 keypoints did not finish the first map's pairs within an hour,
with the GPU shared. The screening stopped there; MatchAnything, MINIMA, UFM, GIM, EfficientLoFTR and RaCo were
not scored.

The graph is limited by how many recordings pass a place, more than by the matcher. sequence_3_18 scores 55–59 in
the eight-recording graph whatever the matcher, and 91 in the test graph.

## Running it

On the GPU machine, with `WORK` holding the recordings' work directories (`WORK/SEQ`) and
`XS_DESC=~/lamaria_xs`, the descriptors on ext4:

```sh
# per test walk with GPS: the keyframes and the own map (the next download can overlap the map)
python scripts/lamaria_xs_keyframes.py $WORK $SEQ $SUB/slam/$SEQ.txt $GPS/$SEQ.csv
python scripts/lamaria_xs_map.py $WORK $SEQ
# the cross edges of all of them, in three runs
python scripts/lamaria_xs_match.py $WORK $WORK/edges_a $SEQS --every 4 --max-maps 2
python scripts/lamaria_xs_match.py $WORK $WORK/edges_b $SEQS --every 4 --offset 2 --max-maps 2
python scripts/lamaria_xs_match.py $WORK $WORK/edges_c $SEQS --every 4 --map-from 2 --max-maps 4
# the pose graph (submission 8's settings are the defaults), the estimates, the checks, the zip
python scripts/lamaria_xs_graph.py $WORK $WORK/solve $SEQS --edges $WORK/edges_{a,b,c}/edges.npy
for s in $TEST; do
  python scripts/basalt_tum_to_lamaria_estimate.py --in-tum $WORK/solve/$s.tum --out-estimate $WORK/slam/$s.txt
done
python scripts/lamaria_gps_agreement.py $SUB/slam $WORK/slam $GPS $TEST
python scripts/lamaria_assemble_submission.py $SUB/submission.zip $WORK/slam submission8.zip $TEST
```

`SEQS` lists the test walks first, then the training walks. The training walks' directories link to the
chain's training work, with every other keyframe of its ALIKED features. The training walks' scores come from
the official evaluator on their converted `.tum` files, as for the chain.

Submission 9 then adds the walks with poor GPS (`LEFT`) and the Moving recordings (`MV`), each with their own
keyframes and maps first:

```sh
# the walks with poor GPS: two matching runs, then a solve with submission 8's 55 held
L=${LEFT// /,}
python scripts/lamaria_xs_match.py $WORK $WORK/left_a $SEQS $LEFT --only $L --loose $L --max-maps 4 \
  --radius 30 --window 5 --every 10
python scripts/lamaria_xs_match.py $WORK $WORK/left_b $SEQS $LEFT --only $L --loose $L --max-maps 4 \
  --place $WORK/left_a/edges.npy --every 2
python scripts/lamaria_xs_graph.py $WORK $WORK/left_hold $SEQS $LEFT --hold $WORK/solve \
  --edges $WORK/edges_{a,b,c}/edges.npy $WORK/left_{a,b}/edges.npy
# the Moving recordings: their pairs only, then a solve with the 58 walks held (HOLD links to both solves)
python scripts/lamaria_xs_match.py $WORK $WORK/mv $SEQS $LEFT $MV --only ${MV// /,} --every 2 --max-maps 4
python scripts/lamaria_xs_graph.py $WORK $WORK/mv_solve $SEQS $LEFT $MV --hold $HOLD \
  --edges $WORK/edges_{a,b,c}/edges.npy $WORK/left_{a,b}/edges.npy $WORK/mv/edges.npy
```

The method description drafted for submission 9 stated the multi-session processing and the GPS:

> Multi-session: 49 test walks with usable GPS, plus six training walks without their ground truth, are refined
> jointly offline in a 4-DoF pose graph, with cross-recording ALIKED+LightGlue localization and per-recording
> odometry weights from one round of variance-component estimation. Three more walks with poor GPS, and 8 of the
> 10 Moving recordings, then join that graph with the others held fixed; the two Moving recordings whose fit to
> GPS got worse keep their earlier estimates. GPS proposes the candidate pairs and places the recordings roughly
> before the solve.
