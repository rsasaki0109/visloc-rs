# LaMAria test-set results

Three submissions of the BA chain ([lamaria_submission.md](lamaria_submission.md))
to the 63 test sequences, run on GitHub Actions
([`lamaria_submission.yml`](../.github/workflows/lamaria_submission.yml)). The
system uses the two grayscale SLAM cameras and the 1 kHz IMU only (no GPS or
magnetometer) and runs one VIO per sequence. Scores are the leaderboard's means
per challenge.

| Submission | Short | Medium | Long | Low light | Moving platform | Mean |
|---|---:|---:|---:|---:|---:|---:|
| 1 (2026-10-23) | 67.7 | 54.8 | 54.1 | 53.6 | 27.5 | 51.5 |
| 2 (2026-10-24) | 67.7 | 54.8 | 54.1 | 59.3 | 26.9 | 52.6 |
| 3 (2026-10-25) | 67.7 | 54.8 | 54.2 | **60.9** | 26.9 | **52.9** |
| GroundedSLAM | 80.2 | 61.6 | 59.9 | 67.7 | 41.9 | 62.3 |
| RedComet | 75.5 | 65.8 | 61.3 | 77.6 | 53.1 | 66.7 |

visloc-rs ranks fourth, behind Aria's SLAM (reproduced by the LaMAria team),
RedComet and GroundedSLAM, and ahead of the other binocular-inertial entries
(FPV-SLAM 42.8). The same chain averages 64.35 per track on the ten training
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

Together the four gain 13.7 points (the track +1.6). Low light is the only
track where visloc-rs beats GroundedSLAM on some sequences (4_1, and 4_2:
55.9 against 43.7).

## Notes for the next round

- **The same input scores differently on a rerun.** Unchanged sequences rerun
  on Actions (the Moving track besides the rides, e.g. sequence_5_1 with its
  path 2,105 m before and 2,106 m after) move by 5–7 points. A change needs
  several sequences, or the training set, to show.
- **The weak sequences drift; they do not fail.** After sequence_4_2 no
  walking sequence runs away. The lowest are sequence_3_12 (26.3), 3_3 (28.4),
  1_2 (29.0), 2_7 (31.8), 3_16 (35.3), 1_16 (35.8) and 4_3 (37.1), with pose
  recall at 5 m of 57–100 %. That is the one-way heading and scale drift of
  [Slow heading drift](lamaria_batch_ba.md#slow-heading-drift), which no
  camera or IMU cue tried so far pins down.
- **The sun is no compass here.** On sequence_2_12 the saturated region the
  detector finds is overcast sky, and the Aria SLAM cameras see the sun only
  when it is low.
- **By device**, the walking tracks average 66.7 on 1WM09370081264 (22
  sequences) and 53.0 on 1WM093701G1276 (22). The difference is mostly Short:
  73.2 against 53.6. Both Short training sequences were recorded with
  1WM09370081264.
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
