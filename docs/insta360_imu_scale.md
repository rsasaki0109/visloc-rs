# Metric scale from a 360 camera's IMU (OverMaps360_50)

A reproduction of Cuttini et al., *arXiv 2610.35554*: the metric scale of an
SfM model built from an Insta360 360 camera, recovered from the camera's own
IMU by a per-run visual-inertial linear alignment. It runs on the
[OverMaps360_50](https://huggingface.co/datasets/OverTheReality/OverMaps360_50)
dataset (50 walks of about 10 minutes each, Insta360 X5, CC BY-NC 4.0; the
paper's "Production B" set). Code:
[`scripts/insta360_imu_scale.py`](../scripts/insta360_imu_scale.py) (one
scene, about 1.5 minutes on a CPU) and
[`scripts/overmaps360_imu_scale.sh`](../scripts/overmaps360_imu_scale.sh)
(all scenes). Nothing in the Rust crates uses this yet.

## Results (50 scenes, 2026-10-28)

Scale error against the ARKit scale of each model (IMU scale over ARKit
scale, minus one), averaged over the two fisheye lenses:

| Estimate | Scenes | Mean | MAE | SD |
| --- | ---: | ---: | ---: | ---: |
| **Per run** (the paper's estimate) | 50 | **−0.21%** | **1.91%** | **2.49%** |
| Joint (one s, g, b_a for every run) | 50 | +0.82% | 2.54% | 3.89% |
| Per run with the angular-velocity term (below) | 50 | +0.73% | 1.86% | 2.45% |
| Per run on every third frame (1.0 s) | 50 | +1.47% | 2.53% | 3.28% |
| Per run on every third frame, with the term | 50 | +1.67% | 2.57% | 3.34% |
| Per run, phone path under 150 images | 42 | −0.81% | 1.65% | 1.92% |
| Per run with the term, phone path under 150 images | 42 | +0.10% | 1.43% | 1.80% |
| Per run, phone path of 150 images or more | 8 | +2.99% | 3.28% | 2.85% |
| Per run with the term, phone path of 150 images or more | 8 | +4.04% | 4.09% | 2.86% |
| Joint, phone path of 150 images or more | 8 | +7.14% | 7.14% | 4.98% |

- **Against the paper**: on the same 50 surveys it reports a mean IMU/ARKit
  ratio of 0.997 with an SD of 2.3%; here 0.998 and 2.5%. 32 scenes are
  within 2%, 49 within 5%, the worst is +8.1% (2ddac689).
- **The paper's estimate reads low**: the model's camera centres carry an
  error proportional to the camera's angular velocity (rolling shutter;
  [Frame spacing](#frame-spacing-an-angular-velocity-error-in-the-camera-centres)).
  Fitting it raises the scale on all 50 scenes, by 0.93 points on average
  (SD 0.44), and the IMU then reads +0.7% above ARKit (+0.1% on the 42
  scenes with short phone captures, with the lowest MAE and SD of any row).
- **Per run against joint**: solving each run separately cuts the SD from
  3.9% to 2.5%, and on the eight scenes with long phone captures the mean
  from +7.1% to +3.0%. The paper finds the same on its ten surveys with an
  external reference (joint +3.39% mean; per run 0.85% MAE).
- **Long phone captures: the ARKit reference reads low**: all eight scenes
  with 150 phone images or more read high against ARKit (+4.0% with the
  angular-velocity term, against +0.1% on the other 42). Against the GPS
  the phone logged during the walk, ARKit reads 3.8% low on them and the
  IMU with the term +0.05% (MAE 0.78%)
  ([Against GPS](#against-gps-the-arkit-reference-on-long-phone-captures)).
  Over 49 scenes the IMU with the term is the estimate closest to the GPS
  (MAE 1.34%, SD 1.52%; ARKit 2.25% and 2.31%).
- **The reference's own error**: on the 15 scenes whose ARKit scale varies by
  more than 3% between quarters of the phone path, the per-run MAE is 2.52%
  against 1.64% on the other 35.
- The two lenses agree to a median 0.73% (at most 1.94%). 1,262 of 1,294
  runs pass the gravity check; the recovered gravity is 9.71–9.85 m/s²
  (median 9.79, as in the paper).

<details>
<summary>Per scene (scene id prefix, phone images in the model, spread of the ARKit scale over the four quarters of the phone path, scale errors against ARKit, then ARKit's and the corrected IMU's against GPS)</summary>

| Scene | Phone images | ARKit quarters spread | Per run | With the term | Every third frame, with the term | Joint | ARKit vs GPS | With the term vs GPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0313afa5 | 400 | 2.0% | +2.09% | +3.30% | +4.50% | +6.57% | −3.57% | −0.38% |
| 0400af24 | 70 | 5.0% | +1.45% | +2.45% | +3.17% | +2.24% | −3.99% | −1.63% |
| 0be9b9d1 | 330 | 4.7% | +4.27% | +5.29% | +7.70% | +6.91% | −4.76% | +0.28% |
| 0c05c38a | 92 | 2.2% | +2.17% | +2.77% | +3.66% | +2.70% | −4.30% | −1.65% |
| 0e55b6bd | 68 | 3.0% | −1.12% | −0.14% | −0.37% | −0.98% | +1.07% | +0.93% |
| 1249636a | 367 | 3.7% | −0.15% | +0.90% | +3.27% | +1.51% | −0.18% | +0.72% |
| 13ed7291 | 54 | 1.7% | −3.90% | −2.51% | −2.43% | −4.00% | +1.47% | −1.08% |
| 1510c3fa | 52 | 1.5% | −1.35% | +0.26% | +0.24% | −1.00% | −1.93% | −1.67% |
| 19c2c02d | 48 | 1.1% | −1.94% | −1.38% | −1.66% | −2.07% | −0.27% | −1.64% |
| 1a751eba | 65 | 2.0% | −0.71% | +0.83% | +2.03% | +0.11% | −0.86% | −0.04% |
| 222f9280 | 47 | 1.5% | −4.21% | −2.43% | −0.41% | −3.27% | −1.15% | −3.55% |
| 27dcbf62 | 63 | 2.0% | −4.34% | −3.38% | −3.90% | −4.12% | +1.73% | −1.71% |
| 2c553d8a | 133 | 91.7% | −1.87% | −0.98% | +0.03% | −1.40% | −1.90% | −2.86% |
| 2ddac689 | 367 | 7.0% | +8.08% | +9.10% | +11.44% | +11.94% | −8.61% | −0.29% |
| 2e776399 | 285 | 3.1% | +4.37% | +5.21% | +6.73% | +15.23% | −4.66% | +0.31% |
| 30012d40 | 67 | 2.8% | −0.05% | +0.96% | +0.30% | −0.10% | −2.83% | −1.89% |
| 312ab70b | 75 | 2.6% | −0.67% | 0.00% | +0.07% | −0.32% | −2.27% | −2.27% |
| 36bc364b | 142 | 4.0% | −3.69% | −2.46% | −0.18% | −2.97% | +2.44% | −0.08% |
| 3886d218 | 119 | 2.6% | −1.32% | +0.11% | +5.59% | −1.49% | +0.66% | +0.77% |
| 3a538821 | 81 | 3.5% | −1.87% | −1.07% | −0.85% | −1.72% | −0.96% | −2.02% |
| 3af6f1f3 | 49 | 2.3% | −2.42% | −0.36% | +0.46% | −1.31% | −2.87% | −3.22% |
| 3b81dbc3 | 66 | 0.9% | −0.69% | −0.27% | −1.13% | −0.80% | −1.79% | −2.06% |
| 3e7efa72 | 109 | 1.4% | −0.50% | +0.41% | +6.05% | +2.41% | +0.83% | +1.24% |
| 43a80210 | 319 | 9.2% | +2.70% | +4.25% | +5.77% | +8.54% | −3.43% | +0.68% |
| 479a12a7 | 58 | 3.0% | −1.84% | −0.97% | +0.40% | −1.19% | +0.08% | −0.89% |
| 488c309b | 58 | 1.5% | +0.09% | +0.87% | −0.72% | −0.02% | −1.03% | −0.17% |
| 4a14bb85 | 53 | 1.1% | −1.76% | −1.14% | −0.69% | −1.24% | −5.51% | −6.59% |
| 51a573ab | 73 | 0.5% | −0.02% | +1.21% | +2.26% | +0.16% | −2.10% | −0.92% |
| 541ca7ec | 59 | 1.2% | −2.93% | −2.79% | −2.85% | −3.18% | +2.04% | −0.81% |
| 5bb3848a | 148 | 8.1% | +0.87% | +2.02% | +2.66% | +0.76% | — | — |
| 5ee6c10b | 49 | 2.3% | −4.66% | −3.24% | −2.03% | −5.57% | +2.54% | −0.79% |
| 68daf868 | 63 | 0.8% | −1.53% | −1.28% | −0.08% | −1.79% | −1.04% | −2.30% |
| 69175f34 | 93 | 0.7% | −0.73% | +0.23% | +0.06% | −0.04% | −1.99% | −1.77% |
| 6f690942 | 74 | 1.7% | +1.79% | +2.65% | +2.67% | +1.86% | −3.57% | −1.01% |
| 7590cf3c | 72 | 3.1% | +1.41% | +1.77% | +1.45% | +1.17% | −4.52% | −2.82% |
| 79e999d1 | 62 | 1.8% | −1.61% | −0.47% | −0.50% | −1.19% | −0.06% | −0.53% |
| 7e14c5e5 | 96 | 1.7% | +3.19% | +4.08% | +4.65% | +3.56% | −3.53% | +0.41% |
| 9529e3cc | 130 | 1.1% | +0.87% | +1.57% | +2.77% | +1.06% | −4.68% | −3.19% |
| 9d2fe9b1 | 71 | 6.1% | −1.00% | −0.24% | −1.33% | +1.65% | +1.16% | +0.91% |
| acc763f0 | 53 | 5.9% | +1.26% | +1.80% | +1.53% | +0.71% | −1.79% | −0.03% |
| ad6f1c8e | 83 | 1.5% | +0.78% | +1.44% | +1.67% | +1.26% | −1.55% | −0.14% |
| b71fe29b | 180 | 1.9% | +3.59% | +4.44% | +6.45% | +6.42% | −3.00% | +1.31% |
| b98b7d85 | 69 | 2.9% | +0.12% | +0.43% | +1.15% | +0.29% | −0.56% | −0.13% |
| c8c675bc | 81 | 0.9% | −2.37% | −1.03% | −2.02% | +0.92% | −0.83% | −1.85% |
| cfa668d0 | 59 | 1.4% | +0.35% | +0.40% | +1.24% | +0.54% | +0.09% | +0.49% |
| d3f7fa64 | 52 | 1.1% | −2.57% | −1.96% | −1.41% | −3.27% | +1.62% | −0.37% |
| dc27af04 | 71 | 1.5% | +0.10% | +0.72% | +1.27% | +0.42% | −1.60% | −0.89% |
| e24912d9 | 42 | 3.1% | +3.01% | +3.34% | +11.47% | +4.77% | −2.03% | +1.24% |
| e5cc9f26 | 197 | 3.0% | −1.03% | −0.20% | +0.03% | +0.03% | −2.05% | −2.25% |
| e8e75511 | 77 | 1.7% | −0.03% | +1.86% | +3.08% | +0.27% | −2.88% | −1.07% |

</details>

## The dataset's IMU CSV holds raw counts

`insta360_imu.csv` labels its columns `gyro_*_dps` and `accel_*_mg`, but the
values are the sensor's raw counts:

- **Gyro**: 16.4 counts per deg/s (a ±2000 dps range). Taken as deg/s, the
  camera seems to turn at a median 174°/s while the model's own rotations
  give about 7°/s, and nothing aligns. Fitted to the model's rotations per
  scene: 16.27 to 16.63 counts per deg/s, median 16.35.
- **Accelerometer**: 1024 counts per g (a ±32 g range). Taken as mg, it
  reads 2.4% high, and the IMU scale comes out 2.4% high with it.

On scene 0313afa5 the trailer at the end of the `.insv` (the last 24 MB)
confirms it: the camera writes 553,216 IMU samples as a 64-bit microsecond
timestamp and six 16-bit counts (offset binary), firmware v1.11.6, and the
CSV is exactly those counts with the axes reordered. The paper reads the
`.insv` with telemetry-parser instead of using the CSV.

The script uses the nominal 16.4 and 1024 for every scene and reports the
measured values only as checks. 1 g read from the specific force averaged
over 20 s windows spans 1001 to 1024 counts across the scenes: the average
shrinks when the camera turns within the window, so it cannot tell 1000
from 1024. An earlier version picked the nearer of the two and chose 1000
on nine scenes, which then recovered |g| ≈ 10.0 m/s² against 9.7–9.9
everywhere else.

## Other facts about the layout

- **Perspective model** (`colmap_perspective/sparse/0`): images are named
  `<2 i + lens>_perspective_<view>.jpg`, five views (0, 1, 2, 4, 5) per
  fisheye lens, so consecutive frame numbers alternate between the two
  lenses, 180° apart and 5.9 cm apart (scene 0313afa5). i is the frame's
  index in time; the script uses view 0 of each lens. The phone images
  (`<sequence>_<frame>.jpg`) are registered in the same model.
- **Frame timing**: every 10th frame of the 29.97 fps video, t_i =
  0.333680 i − 0.335 s on the IMU clock, the same on every scene. The drift
  check (b refitted on the first and last quarters) moves it by under 2 ms
  on 48 scenes and 8.4 ms at most; the paper flags a wrong timing model
  above 60 ms.
- **Camera from IMU**: a 90° axis permutation; the median rotation residual
  of the fit is 0.077° per frame interval (0.03° to 0.6° across scenes;
  three scenes above 0.2°).
- **Frame**: gravity is +Y in the model (the datasheets' `up_sign` −1).
- **Reference**: the model is scaled to the phone's ARKit poses. The script
  fits a Sim(3) from the model's phone images to the ARKit positions. The
  phone path is short (42 to 400 images). Fitted on each quarter of it, the
  ARKit scale spreads by under 3% on 35 scenes and by 3–9% on the rest
  (92% on 2c553d8a), so the reference itself carries about 2% of error (the
  paper: smartphone MAE 1.94% against LiDAR).

## Method

As in the paper (sections 3 and 4), per scene:

1. Frame timing t_i = a i + b and the camera-from-IMU rotation: a coarse
   grid on the frame-to-frame rotation angles, a finer grid, then
   Nelder-Mead on the median residual of the Kabsch fit r_C = R_CB r_B.
2. Runs of frames at most 1.2 s apart with at least 8 intervals; per run, a
   linear least-squares fit of the scale s, gravity g, accelerometer bias
   b_a and one velocity per frame, with IMU preintegration between frames
   (rotation from the model, propagated by the gyro):
   `s (C_{k+1} - C_k) - v_k dt - g dt²/2 + J_k b_a = Δp_k` and
   `v_{k+1} - v_k - g dt + K_k b_a = Δv_k`.
3. Intervals with residuals above five times the median are dropped and the
   run refitted; runs with |g| outside [9.3, 10.3] m/s² are discarded. The
   scene's scale is the mean over the remaining runs. The joint variant fits
   one s, g and b_a to every run at once.

Added here:

- Runs also break where the model's speed exceeds three times its median.
  Scene 0c05c38a has eight misregistered frames (one about 38 m off, a
  115 m/s step); a run containing one fits a scale near 0.01, sometimes with
  a plausible |g|, and the 5×-median rule cannot catch it because the first
  fit is already wrong. Four scenes have such jumps (2 to 10 each).
- R_CB is refitted without the intervals whose rotation residual is above
  three times the median, so misregistered frames do not tilt it.
- Reported next to the paper's estimate: the per-run fit with the
  angular-velocity term of
  [Frame spacing](#frame-spacing-an-angular-velocity-error-in-the-camera-centres),
  and both fits on every third frame.

## Checks on scene 0313afa5

- A synthetic walk (IMU synthesised at 1 kHz, frames at 3 fps) recovers the
  scale to 0.05%;
  [`scripts/tests/test_insta360_imu_scale.py`](../scripts/tests/test_insta360_imu_scale.py)
  runs the whole scene analysis on one.
- Adding an IMU lever arm to the unknowns puts the IMU midway between the
  two lenses (0.019, −0.007, −0.018) m against half the lens offset
  (0.021, −0.006, −0.019), and moves the scale by −0.2 points.
- Shifting the accelerometer by ±10 ms against the gyro raises the fit's
  residual and moves the scale by ±0.3 points.

## Frame spacing: an angular-velocity error in the camera centres

The paper's estimate grows with the frame spacing: on every third frame
(1.0 s) it is higher than on every frame (0.33 s) on 45 of the 50 scenes, by
1.68 points on average (on 0313afa5: +2.3% / +1.9% for lens 0 / 1 at
0.33 s, +4.6% / +4.5% at 1.0 s). The diagnosis, on 0313afa5 and 0c05c38a
with velocity-free second differences of the camera centres over equally
spaced frames, `s (C_{k+m} - 2 C_k + C_{k-m}) = g dt² + q - M b_a`:

- **Only the horizontal components depend on the spacing.** Fitted per
  axis, the vertical scale (mostly the walking bounce) is 1.036–1.047 at
  every spacing on 0313afa5, the horizontal ones 1.00–1.01 at 0.33 s and
  1.04–1.05 at 1.0 s. The horizontal residual is twice the vertical one.
- **Not** the IMU timing (moving the IMU windows for the positions by
  ±20 ms raises the vertical residual from 2 mm to 9–12 mm), a gyro bias
  (0.1°/s at most; removing it moves the scale by 0.05 points), or an IMU
  lever arm (the two lenses, on either side of the IMU, show the same
  dependence; with the lever arm fitted they agree exactly and the
  dependence stays).
- **SfM noise attenuates the fit a little**: regressing the second
  differences on the IMU prediction instead (SfM noise as response noise)
  gives 0.9 points more at 0.33 s and 0.4 at 1.0 s, but still rises with the
  spacing.
- **A camera-centre error proportional to the angular velocity explains
  it**: adding e = R_wc A w_c (A a 3×3 unknown, w_c the camera's angular
  velocity from the gyro) cuts the horizontal residual from 3.8 to 0.9 mm,
  and the scale no longer depends on the spacing (0313afa5, lens 0:
  1.049, 1.047, 1.041 at 0.33, 0.67, 1.0 s; horizontal and vertical
  agree). The fitted A is skew-symmetric, e = R_wc (c × w_c), with c along
  the image's vertical axis: c ≈ (0, −0.014, 0) m·s (median), and on all
  50 scenes c_y is between −0.009 and −0.021 m·s, with the same sign for
  both lenses. An error of this form, set by the camera's rotation and
  oriented along the image rows' readout direction, is what a rolling
  shutter leaves in a global-shutter reconstruction.

In the script, the term is three unknowns per run (c, scaled by s) in the
paper's formulation. With it the scale rises on all 50 scenes (0.93 points
on average, SD 0.44) and the gap between 0.33 s and 1.0 s halves (1.68 to
0.94 points; the rest is mostly the SfM-noise attenuation, which is smaller
at 1.0 s). Every third frame avoids most of both errors but uses a third of
the intervals, so its scatter is larger (SD 3.3% against 2.5%).

## Against GPS: the ARKit reference on long phone captures

The phone logged GPS at 1 Hz throughout the 360 walk
(`backgroundlocationData.txt`, 555 s on 0313afa5, reported accuracy
1.4–2.3 m), an independent metric reference. `gps_scale` fits the log's
time offset (−5 to +2 s) and a 2-D similarity from the model's horizontal
camera centres to GPS east/north. The median residual is 1.3 m (0.5–4.6 m);
on 5bb3848a the fit fails (11.7 m) and that scene is left out. Scale errors
against the GPS scale, 49 scenes:

| Estimate | Scenes | Mean | MAE | SD |
| --- | ---: | ---: | ---: | ---: |
| ARKit | 49 | −1.61% | 2.25% | 2.31% |
| Per run (the paper's estimate) | 49 | −1.88% | 1.97% | 1.60% |
| **Per run with the angular-velocity term** | 49 | **−0.96%** | **1.34%** | **1.52%** |
| ARKit, phone path of 150 images or more | 8 | −3.78% | 3.78% | 2.44% |
| Per run with the term, phone path of 150 images or more | 8 | +0.05% | 0.78% | 1.08% |
| ARKit, phone path under 150 images | 41 | −1.19% | 1.95% | 2.06% |
| Per run with the term, phone path under 150 images | 41 | −1.16% | 1.45% | 1.53% |

- **The long-phone scenes' excess is ARKit's**: on seven of the eight
  scenes with long phone captures ARKit reads 2.0–8.6% below the GPS
  (1249636a: −0.2%), while the IMU with the term is within 1.4% of it on
  all but e5cc9f26 (−2.3%, where ARKit is −2.1%). About 0.8 points of the
  ARKit deficit are the Sim(3) fit itself: over the whole phone path ARKit
  and the model differ by more than a similarity (0.18 m median residual on
  0313afa5 against 0.03 m on 0c05c38a), which shrinks the fitted scale; on
  these eight scenes the median of the four quarters' scales is 0.81% above
  the whole path's. The rest is ARKit's own scale over the longer sessions.
- **Not the model drifting**: the IMU scale of the runs near the phone path
  equals that of the runs elsewhere (0313afa5: 1.0403 and 1.0399).
- **On the other 41 scenes** ARKit and the IMU with the term agree with each
  other (+0.05%) and both read about 1.2% below the GPS. Two independent
  sensors agreeing suggests the phone's GPS scale is about 1% high there
  (1 Hz smartphone fixes in streets; not investigated further).

## Reproduce

```bash
# Accept the dataset's terms on Hugging Face first; HF_TOKEN is a read token.
HF_TOKEN=... scripts/overmaps360_imu_scale.sh ~/om360            # all 50 scenes, ~75 min
HF_TOKEN=... scripts/overmaps360_imu_scale.sh ~/om360 0313afa5-8b3c-4fb1-8245-c6b5abf738d2
python3 scripts/insta360_imu_scale.py summary ~/om360/res/*.json
```

Per scene it downloads about 0.3 GB (the model's `images.bin`, the IMU CSV,
the phone poses and GPS log) and deletes it afterwards. Needs Python with numpy and
scipy.
