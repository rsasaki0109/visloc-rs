# GPS on the LaMAria Moving platform track

Until 2026-11-05 the LaMAria chain used the two SLAM cameras and the IMU only.
Since then it may use any sensor the recordings carry. The leaderboard labels a
method's sensors only as mono, bino and imu, so the method's description states
the GPS. The GPS is used in one place: on the Moving platform track it bridges
the jumps where the trajectory missed a ride or got its length wrong.
Submission 7 does this and takes the track from 30.6 to 47.3. The walks are
left as they are, and the training scores in the README use the cameras and
the IMU only.

## The GPS of the recordings

The raw VRS recordings carry 1 Hz GPS fixes (stream 281-1); the ASL zips the
chain runs on do not. Against the pseudo ground truth of the training
sequences, the fixes are off by 2–8 m at the median and by 5–30 m at the
90th percentile. The worst is 131 m, on sequence_1_19. The errors keep one
direction for about 30 s, and on sequence_1_19 the GPS track is 4 % larger
than the ground truth's.

## Heading from GPS: a negative result

The first use tried was the slow heading drift
([Slow heading drift](lamaria_batch_ba.md#slow-heading-drift)): the drift rate
that fits all fixes, taken out of the trajectory as a rotation growing with
time.

| Sequence | Chain | Heading from GPS | GPS drift rate (°/min) |
|---|---:|---:|---:|
| sequence_1_19 | 82.25 | 69.42 | −0.15 ± 0.37 |
| sequence_1_20 | 88.65 | 81.57 | +0.06 ± 0.07 |
| sequence_2_11 | 89.25 | 77.54 | −0.06 ± 0.07 |
| sequence_3_18 | 54.27 | **60.24** | +0.14 ± 0.06 |
| sequence_4_10 | 59.80 | **61.08** | +0.05 ± 0.04 |
| sequence_4_11 | 71.07 | 62.85 | +0.11 ± 0.03 |
| Mean | **74.22** | 68.78 | |

On the good sequences a rotation of about 1° over the whole sequence costs
7–12 points, and the chain's heading is already better than that. The fixes'
bias in the streets does not show in their spread: on sequence_4_11 the rate
comes out as +0.11 ± 0.03°/min and costs 8 points. Only sequence_3_18, whose
heading is off by several degrees, gains clearly, and nothing without ground
truth tells it from the others.

### What GPS measures of the heading

Per window, the chain's heading against the fixes (a robust 2D similarity in
the window, relative to the whole sequence's) and against the pseudo ground
truth. The second is the chain's own error, and the difference of the two is
the error of measuring it by GPS. Ranges over the six sequences (at 600 s
over five, of three or four windows each; sequence_1_19 is too short):

| Window | Chain's heading error (RMS) | GPS's error in measuring it | Chain's scale error | GPS's error in measuring it |
|---|---:|---:|---:|---:|
| 60 s | 0.1–1.1° | 2.7–6.5° | 0.4–2.2 % | 6.9–14 % |
| 300 s | 0.07–0.86° | 0.8–1.9° | 0.2–1.0 % | 1.5–5.6 % |
| 600 s | 0.09–0.75° | 0.14–1.2° | 0.08–0.6 % | 0.3–1.8 % |

The drift rate through the 300 s windows, in °/min:

| Sequence | By the pseudo ground truth | By GPS |
|---|---:|---:|
| sequence_1_19 | +0.008 | +0.368 |
| sequence_1_20 | −0.008 | −0.184 |
| sequence_2_11 | +0.010 | −0.160 |
| sequence_3_18 | +0.026 | +0.176 |
| sequence_4_10 | +0.094 | +0.078 |
| sequence_4_11 | +0.017 | +0.102 |

On five of the six, GPS's rate is 6–46 times the true one, on two of them
with the other sign. They agree only on sequence_4_10, whose drift is the
largest. sequence_3_18 and 4_11
look alike to GPS (0.10–0.18°/min); both are at 0.02–0.03°/min.

### Taking out only a large drift

A drift can be taken out only when GPS measures it well above its own bias.
With the rate fitted to all fixes at once (the trajectory's steps turned by
k·t, then one robust similarity), the six trajectories as they are measure
−0.02 to +0.22°/min. The test is that of the missed rides: a drift of
0.3, 0.6 or 1.0°/min is turned into the trajectory, and the measured rate is
taken out when it is 0.45°/min or more, twice the largest of the six.

| Sequence | As run | +0.3°/min (left as it is) | +0.6°/min | Taken out | +1.0°/min | Taken out |
|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | 82.25 | 55.67 | 39.17 | measured 0.38, left | 27.41 | 62.79 |
| sequence_1_20 | 88.65 | 48.09 | 34.58 | 79.23 | 22.84 | 79.23 |
| sequence_2_11 | 89.25 | 45.36 | 28.62 | 85.60 | 15.81 | 85.60 |
| sequence_3_18 | 54.27 | 44.23 | 22.20 | 59.96 | 9.53 | 59.96 |
| sequence_4_10 | 59.80 | 43.35 | 23.73 | 61.08 | 10.08 | 61.08 |
| sequence_4_11 | 71.07 | 50.58 | 39.92 | 62.85 | 29.00 | 62.85 |

- A drift of 0.6°/min or more comes back to 60–86 from 10–40. The result does
  not depend on the drift put in: the trajectory lands on the heading GPS
  holds for right, bias included, which is why sequence_1_20 and 4_11 end
  below their scores as run.
- A drift of 0.3°/min already costs 10–44 points, and GPS cannot tell it from
  its bias.

On the walking test sequences (no ground truth involved), the measured rates
stay within ±0.37°/min but for three. sequence_4_9 measures −0.47 ± 0.21
(538 fixes); sequence_1_11 and 1_16 sit at the edge of the search with 42
and 12 fixes. The lowest-scoring walks are below the threshold: sequence_1_2
at −0.34, 3_3 at −0.24 and 3_12 at −0.14°/min. No submission was made from
this: the rule would move one sequence, by a rate barely over the threshold.

## Where GPS helps: jumps of tens of metres

The evaluator fits the trajectory to the control points with a robust Sim(3)
and then refines the fit by plain least squares over all of them. One control
point off by 50 m therefore spoils the others. On the Moving platform track
that happens when the VIO misses a ride or gets its length wrong. The
trajectory keeps its shape, but everything after the ride is displaced by
tens to hundreds of metres. GPS sees such a jump even with an error of
several metres.

The training rides cannot test this. Both happen before the first fix with an
accuracy of 10 m or better:
- sequence_5_12 rides at 83–166 s; its first such fix comes at 273 s.
- sequence_5_11 rides at 119–188 s; its first such fix comes at 671 s.

Their largest errors are at the control points around the rides: 80 m on
sequence_5_12's first and 34 m on 5_11's first.

## The method

[`scripts/lamaria_gps_steps.py`](../scripts/lamaria_gps_steps.py) does not
know where the rides are.
- **Model.** Against the fixes with an accuracy of 10 m or better, the
  trajectory's x, y fits as pieces that share one rotation and one scale but
  not their translation: en ≈ c R xy + t(t). The translation t is constant on
  each piece and changes linearly over a ramp between two pieces (the ride).
  The fit is a least squares with a Cauchy loss of 5 m.
- **Search.** Ramps are added one at a time. In each piece between the ramps
  so far, the script tries a centre every 10 s and a length of 0, 30, 60, 90,
  120, 180 or 240 s, and fits the whole model again for each. It takes the
  ramp that lowers the cost most, if:
  - the levels on its two sides differ by 25 m or more;
  - each side keeps 60 s and 30 fixes.
- **Bridging.** Each ramp's change of level, taken back to the trajectory's
  frame, is spread linearly over the ramp and added to every position after
  it. Only x and y move; the timestamps, z and the rotations stay.

The 25 m are twice the largest step the search finds on the eight training
trajectories as they are. Those have no step to find, and the largest
candidate is 12.6 m, on sequence_1_19.

## Missed rides on the training sequences

To test the bridging on rides it does not know, a trajectory is made to stand
still for 90 s of a walk and to carry on from there displaced by the 50–97 m
it missed. The script is not told where. Scores from the official evaluator:

| Sequence | As run | 90 s missed | Bridged by GPS |
|---|---:|---:|---:|
| sequence_2_11 | 89.25 | 5.55 | 57.26 |
| sequence_1_20 | 88.65 | 6.35 | 47.79 |
| sequence_4_11 | 71.07 | 9.21 | 68.67 |
| sequence_4_10 | 59.80 | 9.00 | 57.32 |
| sequence_3_18 | 54.27 | 6.16 | 43.78 |
| sequence_5_11 | 51.46 | 9.24 | 45.22 |
| sequence_5_12 | 5.80 | 0.00 | 4.53 |

The search finds all seven missed stretches and gives back nearly the
distance missed (71.2 m of 72 m, for one). The scores come back only part of
the way, to 44–69 from 0–9; over the ramp itself the bridged trajectory is a
straight line at constant speed. The eight trajectories as they are come out
unchanged.

## Fetching the GPS without the images

The ten Moving test recordings are 10–31 GB each, 196 GB in all.
[`scripts/lamaria_vrs_fetch.py`](../scripts/lamaria_vrs_fetch.py) fetches
only what a VRS reader needs to open the file and read the GPS. Aria
recordings keep the index of every record in the file's head, before the
first user record (4–30 MB). From that index the script takes these records:
- every configuration and state record;
- the first and the last data record of every stream (the reader looks for
  the last one from the end of the file when it opens it);
- all data records of the GPS, UTC time and time sync streams (281-1, 285-1,
  286-1).

They come with multi-range HTTP requests, 150 ranges a request, and go at
their offsets into a sparse file of the full size.
[`scripts/lamaria_gps_extract.py`](../scripts/lamaria_gps_extract.py) then
reads that file with projectaria_tools like the whole recording. A recording
comes down to 6–35 MB: 0.25 GB for the ten Moving sequences and 1.1 GB for the
53 walks. On training sequence_5_11, a sparse copy made from the whole file
gives the same fixes as the whole file.

```bash
S=sequence_5_5
python scripts/lamaria_vrs_fetch.py https://cvg-data.inf.ethz.ch/lamaria/raw_data/test/$S.vrs work/$S/$S.vrs
python scripts/lamaria_gps_extract.py work/$S/$S.vrs work/gps/$S.csv
python scripts/lamaria_gps_steps.py work/gps/$S.csv slam/$S.txt bridged/$S.txt
```

`lamaria_vrs_fetch.py` needs the `zstd` command for the index. The test runs
kept the sparse files on ext4, where a file of the full size takes up only the
bytes fetched.

## Test set: submission 7

Submission 6's estimates, through the script with its defaults. The residual
is the median distance to the fixes after the fit, with one piece and with the
pieces of the ramps found. The scores are the leaderboard's.

| Sequence | Fixes | Ramps found | Moved | Residual, one piece → bridged | Largest other candidate | Score, submission 6 → 7 |
|---|---:|---|---:|---:|---:|---:|
| sequence_5_1 | 2248 | 153–183 s | 26.7 m | 3.92 → 3.88 m | 7.9 m | 43.1 → 43.4 |
| sequence_5_2 | 1913 | 80–200 s | 70.5 m | 3.88 → 3.72 m | 11.5 m | 44.2 → **51.7** |
| sequence_5_3 | 2192 | 2112–2352 s | 24.1 m | 4.59 → 3.90 m | 11.8 m | 29.3 → **31.0** |
| sequence_5_4 | 1497 | 69–159 s | 308.3 m | 4.82 → 4.52 m | 18.4 m | 39.2 → 36.2 |
| sequence_5_5 | 1865 | 655–715 s | 253.9 m | 5.02 → 4.12 m | 10.6 m | 3.1 → **63.7** |
| sequence_5_6 | 1967 | 139–259, 349–589, 854–884 s | 27.5, 36.8, 244.4 m | 39.56 → 4.32 m | 18.6 m | 2.4 → **35.2** |
| sequence_5_7 | 995 | 119–149 s | 153.8 m | 3.41 → 3.03 m | 5.6 m | 29.6 → **69.2** |
| sequence_5_8 | 1165 | none | | 2.97 m | 8.7 m | 72.3 |
| sequence_5_9 | 1419 | 210–450 s | 47.3 m | 7.00 → 3.60 m | 14.8 m | 10.1 → **37.6** |
| sequence_5_10 | 1014 | none | | 2.79 m | 8.3 m | 32.8 |
| Track | | | | | | 30.6 → **47.3** |

The track gains 16.7 points and the mean 3.3 (54.3 → 57.7). Seven of the
eight bridged sequences gain; sequence_5_4 loses 3.0. The two missed trams
gain the most: 5_5 by 60.6 and 5_6 by 32.8.

- **The rides known before.** The ramps meet the rides found without GPS:
  - sequence_5_5's and 5_6's largest ramps fall in the tram standstills the
    accelerometer had shown, at 10.5–12.7 and 14.0–15.5 min ([Finding the
    rides the VIO missed](lamaria_batch_ba.md#finding-the-rides-the-vio-missed)).
  - sequence_5_4's falls in its ride at 50–180 s, which submission 4
    shortened from 600 m to 214 m.
  - sequence_5_9's falls around its ride at 5.2–7.0 min.
- **sequence_5_10** has no fix before 7.8 min, after its ride at 56–162 s.
  It has no fix of any accuracy before 343 s, so nothing could bridge that
  ride.
  The other nine have fixes from their first 40 s.
- **Close to the threshold.** sequence_5_1's step is 28.0 m and 5_3's 25.1 m.
  Whether they are rides or a stretch of biased fixes is not known; their
  scores move by +0.3 and +1.7.
- **The walks.** On the 53 walking sequences the search finds no step of
  25 m or more; the largest candidate is 24.9 m, on sequence_4_8.
  - sequence_1_17, 2_10 and 4_7 have no fix with an accuracy of 10 m or
    better.
  - At 15 m, sequence_4_8, 4_5 and 3_3 would move by 16–22 m. Over those
    ramps the fixes are sparse and the trajectory drifts away from them
    slowly, without a jump.

Submission 7 is submission 6 with the eight bridged sequences replaced
([lamaria_test_results.md](lamaria_test_results.md)); its 53 walks score as
in submission 6. No test ground truth was used: the threshold and the options
were set on the training sequences.

The method's description on the leaderboard states the GPS (as saved on
2026-11-10):

> Stereo visual-inertial odometry (Rust port of Basalt) followed by an offline
> batch visual-inertial bundle adjustment with loop closure (SIFT features,
> union-find tracks, pose-graph loop correction; on some sequences, ALIKED
> keypoints matched by LightGlue in the bundle adjustment pass). Uses the two
> grayscale SLAM cameras and the 1 kHz IMU. On the Moving platform track only,
> the 1 Hz GPS of the raw recordings bridges jumps: where the trajectory jumps
> by 25 m or more against the GPS fixes (accuracy ≤ 10 m), the pieces between
> the jumps are moved to the GPS. Single VIO run per sequence.
