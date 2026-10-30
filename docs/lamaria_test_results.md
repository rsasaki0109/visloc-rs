# LaMAria test-set results

Six submissions of the BA chain ([lamaria_submission.md](lamaria_submission.md))
to the 63 test sequences, run on GitHub Actions
([`lamaria_submission.yml`](../.github/workflows/lamaria_submission.yml)); the
fifth and the sixth reran five Short sequences locally, on an 8 GB GPU. The
system uses the two grayscale SLAM cameras and the 1 kHz IMU only (no GPS or
magnetometer) and runs one VIO per sequence. Scores are the leaderboard's means
per challenge.

| Submission | Short | Medium | Long | Low light | Moving platform | Mean |
|---|---:|---:|---:|---:|---:|---:|
| 1 (2026-10-23) | 67.7 | 54.8 | 54.1 | 53.6 | 27.5 | 51.5 |
| 2 (2026-10-24) | 67.7 | 54.8 | 54.1 | 59.3 | 26.9 | 52.6 |
| 3 (2026-10-25) | 67.7 | 54.8 | **54.2** | **60.9** | 26.9 | 52.9 |
| 4 (2026-10-27) | 68.9 | **56.8** | 54.1 | 60.6 | **30.6** | 54.2 |
| 5 (2026-10-28) | **70.1** | **56.8** | 54.1 | 60.6 | **30.6** | **54.4** |
| 6 (2026-10-30) | 69.7 | **56.8** | 54.1 | 60.6 | **30.6** | 54.3 |
| GroundedSLAM (2026-10-29) | 87.0 | 72.7 | 70.8 | 81.9 | 49.8 | 72.4 |
| RedComet | 75.5 | 65.8 | 61.3 | 77.6 | 53.1 | 66.7 |
| FPV-SLAM, binocular (2026-10-28) | 77.3 | 58.8 | 51.0 | 64.6 | 29.6 | 56.3 |

visloc-rs ranks fifth on 2026-10-31, behind Aria's SLAM (reproduced by the
LaMAria team), GroundedSLAM, RedComet and the binocular FPV-SLAM. Their last
uploads raised GroundedSLAM's mean from 62.3 to 72.4 and FPV-SLAM's from 42.8
to 56.3. The same chain averages 64.35 per track on the ten training
sequences. Each track's first training sequence is the one the method was
developed on, and the test set behaves like the second ones
([Two sequences per track](lamaria_batch_ba.md#two-sequences-per-track)).

## What changed between the submissions

**Submission 2.**
- **sequence_4_2** went from 4.2 to 55.9 (CP recall at 1 m 0 → 37.5 %, pose
  recall at 5 m 8.3 → 94.1 %). Its VIO, started 10 s into the recording after
  two divergent runs, had run away for 60 s at up to 12 m/s, below the old
  divergence check. A walking track's VIO that averages more than 4 m/s over
  10 s now counts as diverged, and the driver starts it later
  ([`runaway`](../scripts/run_lamaria_test_submission.py); here 60 s in, so the
  first minute has no poses).
- **The Moving track**, rerun with the platform filter
  ([`platform.rs`](../pipelines/basalt/src/platform.rs)), went from 27.5 to
  26.9. sequence_5_10 gains 6.7: the ride from 56 s to 162 s grows from 34 m
  to 145 m. The other sequences move by −7.4 … +5.2 with nearly the same
  trajectories (path lengths within 1 %). On sequence_5_5, 5_6 and 5_9
  (0.3–1.0) the tram is seen only through a door's windows, too few street
  tracks last for the filter to turn on, and the VIO keeps the cabin's
  standstill.

**Submission 3.** The Low light sequences as bright as the day training
sequences (a mean cam0 grey level of 70 or more) ran the day options:

| Sequence | Grey level | Low-light options | Day options | GroundedSLAM |
|---|---:|---:|---:|---:|
| sequence_4_1 | 74.0 | 43.4 | **65.6** | 60.3 |
| sequence_4_4 | 75.0 | 51.6 | **57.3** | 57.5 |
| sequence_4_8 | 74.0 | **69.1** | 57.0 | 84.4 |
| sequence_4_9 | 74.9 | **71.1** | 69.0 | 76.1 |

Together the four gain 13.7 points (the track +1.6), and visloc-rs now beats
GroundedSLAM on sequence_4_1 as on 4_2 (55.9 against 43.7).

**Submission 4.** All 63 sequences again, from main at
[`247d3ff`](https://github.com/rsasaki0109/visloc-rs/commit/247d3ff82db21001daaf5433e43cb533f25a15c5),
with two changes: by day the chain matches the keyframes once more, guided by
the pass ([Matching again after the
pass](lamaria_batch_ba.md#matching-again-after-the-pass)), and the Moving
track's VIO tracks both cameras' features
([lamaria_submission.md](lamaria_submission.md)). The mean goes from 52.9 to
54.2, and 44 sequences move by 1 point or less.
- **By day the trajectories hardly move.** After a Sim(3) fit, 51 of the 53
  sequences off the Moving track come out within 0.3 m (RMS) of submission 3's,
  and their scores move by −2.8 … +3.6. The other two, sequence_2_2 (40.2 →
  62.2) and 1_13 (50.3 → 64.5), already differ by 3–4 % in scale before the
  re-match: the Short and Medium tracks had last run for submission 1. They
  make 0.6 of the mean.
- **The Moving track** goes from 26.9 to 30.6:

  | Sequence | Ride | Ride path, submission 3 → 4 | Score, submission 3 → 4 |
  |---|---|---:|---:|
  | sequence_5_9 | 5.2–7.0 min | 140 → 447 m | 1.0 → 10.1 |
  | sequence_5_10 | 56–162 s | 144 → 353 m | 41.2 → 32.8 |
  | sequence_5_4 | 50–180 s | 600 → 214 m | 39.4 → 39.2 |

  The second camera changes the VIO on the walks as well: after
  sequence_5_10's ride the two trajectories still differ by 3 % in scale and
  4° in heading, and sequence_5_8 (+18.0) and 5_3 (+7.0) gain with
  trajectories within 8 m (RMS) of submission 3's. On sequence_5_4 the second camera's own features cost the first
  ride (on a window of its first four minutes, cam0's features only: 705 m;
  with cam1's detection, landmarks and keyframe links added: 115 m), but a
  ride in the first three minutes costs the score little.

**Submission 5.** The five Short sequences recorded with device
1WM093701G1276 ran again, with ALIKED keypoints matched by LightGlue in the
chain's pass and SIFT online (`run_lamaria_test_submission.py
--ba-pass-features aliked`;
[Learned features](lamaria_batch_ba.md#learned-features)). The other 58
sequences keep submission 4's estimates. They ran locally on an 8 GB GPU, with
the VIO built from `3d6f7530`, the same VIO code as submission 4's.
- **Selection.** On the training set the pass with ALIKED gains only on the
  two Short sequences of that device, R_11_5cp (+6.4) and R_12_10cp (+9.3). It
  loses 1.2–12.7 on sequence_1_19, 1_20, 2_11 and 3_18. On the test set that
  device's Short sequences averaged 57.2 in submission 4, against 73.4 on the
  other device. The serials were read from the recordings' headers:
  sequence_1_1, 1_3, 1_4, 1_13 and 1_18.
- **Change.** Against submission 4's estimates (one Sim(3) fit each, no ground
  truth), the five trajectories move by 0.43–1.14 m (RMS) after the fit. They
  also come out 0.35–0.97 % shorter, as on the training sequences, but the
  evaluation's own similarity fit takes that scale out.
- **Result.** The five average 61.5 instead of 57.2: Short goes from 68.9 to
  70.1 and the mean from 54.2 to 54.4. sequence_1_13 goes from 64.5 to 61.8,
  so the four others gain about 24 points together. Their scores are now 55.2
  (1_1), 63.8 (1_3), 55.4 (1_4) and 71.4 (1_18).
- **A repeat.** Submission 5's archive, uploaded again by mistake on
  2026-10-29, scored the same on all 63 sequences: the evaluation is
  deterministic.

**Submission 6.** The same five sequences again, with the pass's ALIKED
keypoints matched within the online result's epipolar band instead of the
VIO's ([Learned matches without the VIO's
band](lamaria_batch_ba.md#learned-matches-without-the-vios-band)). The run
reuses submission 5's online result, ALIKED keypoints, revisits, IMU and VIO.
- **Selection.** On the device's two Short training sequences the online
  result's band scored 88.17 on average, against 83.08 with the VIO's
  (R_11_5cp 97.02, R_12_10cp 79.31).
- **Change.** Against submission 5's estimates the five trajectories move by
  0.08–0.49 m (RMS) after a Sim(3) fit.
- **Result.** Four of the five lose: sequence_1_1 goes from 55.2 to 54.7, 1_3
  from 63.8 to 61.2, 1_4 from 55.4 to 51.7 and 1_13 from 61.8 to 59.3, while
  1_18 gains (71.4 → 73.0). The five average 60.0 instead of 61.5, Short goes
  from 70.1 to 69.7 and the mean from 54.4 to 54.3. The two training
  sequences did not carry over to the five test sequences; the next
  submission takes submission 5's estimates for them back.

## Against GroundedSLAM, sequence by sequence

GroundedSLAM's scores here are those before its upload of 2026-10-29.

The leaderboard's per-sequence scores (2026-10-27, submission 4): visloc-rs
beats GroundedSLAM on 21 of the 63 sequences, 8 of them Long. The 14
sequences it trails by more than 20 points make up 5.9 of the 8.1 points
between the means (each sequence weighs 1 / (5 × its track's count)); the
other 49 add up to 2.2. By track, the gap's share of the mean: Short 2.27,
Moving 2.26, Low light 1.42, Long 1.16, Medium 0.97. The largest single ones
are sequence_5_9 (10.1 against 60.6, a tram seen through a door's windows:
1.0 point of the mean), 4_8 (57.4 against 84.4; the day options cost it 12
points, see above), 1_2, 4_3, 1_16 and 3_3.

On Short, 6 of the 18 sequences trail by more than 20: sequence_1_2, 1_3,
1_4, 1_11, 1_16 and 1_17 (in submission 3 also 1_1 and 1_13). They are the
fast walks along open streets: those eight walk at 1.1–1.4 m/s, 270–670 m
across, against mostly 0.75–1.16 m/s and 160–280 m for the ten close to
GroundedSLAM (the score correlates −0.59 with the path length,
GroundedSLAM's −0.45). Most keep pose recall at 5 m near 100 % but put few
CPs within 1 m, the slow scale and heading drift of a long one-way walk with
distant features. Training R_12_10cp is such a walk, and [Matching again
after the pass](lamaria_batch_ba.md#matching-again-after-the-pass) takes it
from 61.09 to 64.30; on these test sequences it moved the scores by −0.4 …
+2.6.

## Notes for the next round

- **A rerun repeats a trajectory; a change of the VIO's config moves all of
  it.** With the same inputs and config, 51 of the 53 sequences off the Moving
  track came out within 0.3 m of their earlier run in submission 4. A new VIO
  config changes the walks too, even one meant for the rides: on the Moving
  track sequences moved by 5–7 points with the platform filter where their
  rides did not change (sequence_5_1, its path 2,105 m before and 2,106 m
  after), and by −8.4 … +18.0 with the second camera. A change needs several
  sequences, or the training set, to show.
- **The weak sequences drift; they do not fail.** After sequence_4_2 no
  walking sequence runs away. The lowest are sequence_3_12 (25.9), 3_3 (28.5),
  1_2 (29.0), 2_7 (32.0), 3_16 (35.4), 1_16 (35.7) and 4_3 (37.0), with pose
  recall at 5 m of 55–100 %. That is the one-way heading and scale drift of
  [Slow heading drift](lamaria_batch_ba.md#slow-heading-drift), which no
  camera or IMU cue tried so far pins down.
- **The IMU shows the rides the VIO missed but cannot bridge them.**
  sequence_5_5 and 5_6 (3.1, 2.4) keep a tram's standstill over 10.5–12.7 min
  and 14.0–15.5 min: there the accelerometer's velocity change over 10 s
  windows disagrees with the trajectory's by 1–3 m/s, against 0.2–0.3 m/s on
  the walks. Dead reckoning across a ride is 18 % short to 68 % long on the
  two training rides, so the chain does not bridge them ([Finding the rides
  the VIO missed](lamaria_batch_ba.md#finding-the-rides-the-vio-missed)).
- **The sun is no compass here.** On sequence_2_12 the saturated region the
  detector finds is overcast sky, and the Aria SLAM cameras see the sun only
  when it is low.
- **By device**, the walking tracks average 66.7 on 1WM09370081264 (22
  sequences) and 54.8 on 1WM093701G1276 (22) in submission 4. The difference
  is mostly Short: 73.4 against 57.2 (61.5 in submission 5). The two
  `sequence_1_*` training sequences were recorded with 1WM09370081264, and the
  two R sequences with 1WM093701G1276.
- **The end of sequence_5_12** (21–50 % too long between its last CPs) is
  walking, not a ride. Its CP errors grow from both ends of the recording (55 m
  at the start, 35 m at the end, 5–10 m in the middle), and one CP (8474) sits
  about 10 m aside.

## Running a submission

GitHub Actions → *LaMAria submission* → *Run workflow* on `main`:
- **Split**: test.
- **Sequences**: the ones to rerun, or empty for all 63.
- **Driver arguments**: `driver_args`, if any.

The package job zips the run's estimates.

Timing:
- A sequence takes 35–90 minutes. A VIO that diverges and restarts several
  times takes longer: sequence_4_2's five starts took 2 h 45 min.
- Ten Moving sequences at `max_parallel` 8 took about 2 hours.

To build a full submission, take the previous zip and replace the rerun
sequences. Uploading to <https://lamaria.ethz.ch> stays manual:
- Submit under the existing method; a second test submission for the same
  method is not allowed.
- An update asks for a short reason.
- A new test evaluation is blocked for 24 hours after the previous one
  completed.
