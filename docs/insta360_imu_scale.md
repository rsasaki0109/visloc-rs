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
| Per run, phone path under 150 images | 42 | −0.81% | 1.65% | 1.92% |
| Per run, phone path of 150 images or more | 8 | +2.99% | 3.28% | 2.85% |
| Joint, phone path of 150 images or more | 8 | +7.14% | 7.14% | 4.98% |

- **Against the paper**: on the same 50 surveys it reports a mean IMU/ARKit
  ratio of 0.997 with an SD of 2.3%; here 0.998 and 2.5%. 32 scenes are
  within 2%, 49 within 5%, the worst is +8.1% (2ddac689).
- **Per run against joint**: solving each run separately cuts the SD from
  3.9% to 2.5%, and on the eight scenes with long phone captures the mean
  from +7.1% to +3.0%. The paper finds the same on its ten surveys with an
  external reference (joint +3.39% mean; per run 0.85% MAE).
- **Long phone captures**: all eight scenes with 150 phone images or more
  read high, and the per-run error correlates with the number of phone
  images (r = 0.58). Whether the reference (the model anchored to a longer
  ARKit session) or the model causes it is open.
- **The reference's own error**: on the 15 scenes whose ARKit scale varies by
  more than 3% between quarters of the phone path, the per-run MAE is 2.52%
  against 1.64% on the other 35.
- The two lenses agree to a median 0.73% (at most 1.94%). 1,262 of 1,294
  runs pass the gravity check; the recovered gravity is 9.71–9.85 m/s²
  (median 9.79, as in the paper).

<details>
<summary>Per scene (scene id prefix, phone images in the model, spread of the ARKit scale over the four quarters of the phone path, scale errors)</summary>

| Scene | Phone images | ARKit quarters spread | Per run | Joint |
| --- | ---: | ---: | ---: | ---: |
| 0313afa5 | 400 | 2.0% | +2.09% | +6.57% |
| 0400af24 | 70 | 5.0% | +1.45% | +2.24% |
| 0be9b9d1 | 330 | 4.7% | +4.27% | +6.91% |
| 0c05c38a | 92 | 2.2% | +2.17% | +2.70% |
| 0e55b6bd | 68 | 3.0% | −1.12% | −0.98% |
| 1249636a | 367 | 3.7% | −0.15% | +1.51% |
| 13ed7291 | 54 | 1.7% | −3.90% | −4.00% |
| 1510c3fa | 52 | 1.5% | −1.35% | −1.00% |
| 19c2c02d | 48 | 1.1% | −1.94% | −2.07% |
| 1a751eba | 65 | 2.0% | −0.71% | +0.11% |
| 222f9280 | 47 | 1.5% | −4.21% | −3.27% |
| 27dcbf62 | 63 | 2.0% | −4.34% | −4.12% |
| 2c553d8a | 133 | 91.7% | −1.87% | −1.40% |
| 2ddac689 | 367 | 7.0% | +8.08% | +11.94% |
| 2e776399 | 285 | 3.1% | +4.37% | +15.23% |
| 30012d40 | 67 | 2.8% | −0.05% | −0.10% |
| 312ab70b | 75 | 2.6% | −0.67% | −0.32% |
| 36bc364b | 142 | 4.0% | −3.69% | −2.97% |
| 3886d218 | 119 | 2.6% | −1.32% | −1.49% |
| 3a538821 | 81 | 3.5% | −1.87% | −1.72% |
| 3af6f1f3 | 49 | 2.3% | −2.42% | −1.31% |
| 3b81dbc3 | 66 | 0.9% | −0.69% | −0.80% |
| 3e7efa72 | 109 | 1.4% | −0.50% | +2.41% |
| 43a80210 | 319 | 9.2% | +2.70% | +8.54% |
| 479a12a7 | 58 | 3.0% | −1.84% | −1.19% |
| 488c309b | 58 | 1.5% | +0.09% | −0.02% |
| 4a14bb85 | 53 | 1.1% | −1.76% | −1.24% |
| 51a573ab | 73 | 0.5% | −0.02% | +0.16% |
| 541ca7ec | 59 | 1.2% | −2.93% | −3.18% |
| 5bb3848a | 148 | 8.1% | +0.87% | +0.76% |
| 5ee6c10b | 49 | 2.3% | −4.66% | −5.57% |
| 68daf868 | 63 | 0.8% | −1.53% | −1.79% |
| 69175f34 | 93 | 0.7% | −0.73% | −0.04% |
| 6f690942 | 74 | 1.7% | +1.79% | +1.86% |
| 7590cf3c | 72 | 3.1% | +1.41% | +1.17% |
| 79e999d1 | 62 | 1.8% | −1.61% | −1.19% |
| 7e14c5e5 | 96 | 1.7% | +3.19% | +3.56% |
| 9529e3cc | 130 | 1.1% | +0.87% | +1.06% |
| 9d2fe9b1 | 71 | 6.1% | −1.00% | +1.65% |
| acc763f0 | 53 | 5.9% | +1.26% | +0.71% |
| ad6f1c8e | 83 | 1.5% | +0.78% | +1.26% |
| b71fe29b | 180 | 1.9% | +3.59% | +6.42% |
| b98b7d85 | 69 | 2.9% | +0.12% | +0.29% |
| c8c675bc | 81 | 0.9% | −2.37% | +0.92% |
| cfa668d0 | 59 | 1.4% | +0.35% | +0.54% |
| d3f7fa64 | 52 | 1.1% | −2.57% | −3.27% |
| dc27af04 | 71 | 1.5% | +0.10% | +0.42% |
| e24912d9 | 42 | 3.1% | +3.01% | +4.77% |
| e5cc9f26 | 197 | 3.0% | −1.03% | +0.03% |
| e8e75511 | 77 | 1.7% | −0.03% | +0.27% |

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
- **Open**: the scale depends on the frame spacing. Thinning the frames to
  every second or third (0.67 s, 1.0 s) moves the error from +2.3% / +1.9%
  (lens 0 / 1) to +4.3% / +4.1% and +4.6% / +4.5%.

## Reproduce

```bash
# Accept the dataset's terms on Hugging Face first; HF_TOKEN is a read token.
HF_TOKEN=... scripts/overmaps360_imu_scale.sh ~/om360            # all 50 scenes, ~75 min
HF_TOKEN=... scripts/overmaps360_imu_scale.sh ~/om360 0313afa5-8b3c-4fb1-8245-c6b5abf738d2
python3 scripts/insta360_imu_scale.py summary ~/om360/res/*.json
```

Per scene it downloads about 0.3 GB (the model's `images.bin`, the IMU CSV,
the phone poses) and deletes it afterwards. Needs Python with numpy and
scipy.
