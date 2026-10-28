# visloc-wasm: live VIO in a phone browser

The Basalt port's monocular-inertial VIO (`pipelines/basalt`), compiled to
WebAssembly and driven by a phone's camera and motion sensors in a web page.
It runs on iPhone (Safari) and Android (Chrome; Firefox in a lighter
setting). Everything runs on the phone.

- `src/vio.rs`: the session. IMU samples and grey frames go in as they
  arrive, a pose per frame comes out. It also undistorts frames when the rig
  has lens distortion, and runs a watchdog that restarts a diverged estimate
  and re-anchors it to the last good pose.
- `src/gpu_flow.rs`, `src/flow.wgsl`: the frontend on the GPU (WebGPU in the
  browser, Vulkan or Metal natively); see "The frontend on the GPU".
- `src/sync.rs`: online camera-IMU time offset (and focal length) estimate:
  visual frame-to-frame rotation from the tracked points against the
  gyroscope.
- `src/recording.rs`: the `.vrec` format the page records (frames, IMU,
  rig).
- `src/web.rs`: the wasm-bindgen API (`Vio`), used by `web/worker.js`.
- `web/`: the page. `app.js` handles the sensors, the AR overlay and the
  map. `profiles.js` is the per-phone calibration table. `worker.js` owns
  the session.
- `examples/advio.rs`: replays an [ADVIO](https://github.com/AaltoVision/ADVIO)
  iPhone recording the way a browser feeds it (30 fps, 360 px, IMU resampled
  to 60 Hz) and scores it against ground truth and ARKit.
- `examples/replay.rs`: replays a `.vrec`, for example a phone capture
  uploaded from the page.
- `examples/gpu_check.rs`: compares the GPU frontend with the CPU one on a
  `.vrec`, point by point.

## Build and serve

```bash
./build.sh             # web/pkg (1 thread) and web/pkg-mt (wasm threads, nightly)
python3 serve.py       # http://127.0.0.1:8787 with COOP/COEP, byte ranges, /upload
cloudflared tunnel --url http://127.0.0.1:8787   # https for the phone
```

`build.sh` needs the `wasm32-unknown-unknown` target on the pinned toolchain
and on the nightly (with `rust-src`), and `wasm-bindgen-cli` at Cargo.lock's
version. `wasm-opt` is used if it is on PATH. Camera and motion sensors
need https (or localhost). Threads need the page cross-origin isolated,
which `serve.py`'s headers provide; without them the page falls back to the
single-threaded build. A tab the phone resumes keeps the code it loaded, so
when the page comes back to the foreground with the served files changed,
it reloads itself.

The page also works as a static site: `./deploy_pages.sh <remote>` pushes
it, with both builds and the demo if `web/demo` exists, to a `gh-pages`
branch for GitHub Pages. Static hosts cannot send COOP/COEP, so a service
worker (`web/coi-sw.js`) adds them in the browser, after one reload on the
first visit. Telemetry and uploads need `serve.py` and are off elsewhere:
the page looks for the `X-Visloc-Server` header that `serve.py` sends.

The sensor-free demo (`web/demo/`, not in git) is built from an ADVIO
sequence (CC BY-NC 4.0):

```bash
python3 tools/make_advio_demo.py ~/datasets/advio/advio-02 web/demo/advio-02 --seconds 60
```

`web/vendor/three.js` is three.js r186 plus `OrbitControls`, bundled with
esbuild (only the classes the page uses).

## Calibration per phone

`profiles.js` identifies the phone and loads its numbers.

- **Android:** Chrome reports the exact model through User-Agent Client
  Hints (e.g. `Pixel 8`, `SM-N986U` for a Galaxy Note20 Ultra, `A069P` for
  a Nothing Phone (4a) Pro). The UA string itself no longer names it
  (Chrome says `K`, Firefox `Mobile`), so Firefox phones are unmatched.
- **iPhone:** iOS hides the model, so the phone is matched to a family by
  its screen size and pixel ratio.

The page needs the main camera, the lens the profile describes. On an
iPhone it takes "Back Camera" over the virtual multi-camera devices. On
Android, back cameras are named "camera <id>, facing back" (Chrome) or
"Facing back:<id>" (Firefox), and the page takes the lowest id: Chrome's
default on a Galaxy Note20 Ultra was "camera 2", the 13 mm ultra-wide
(the profile says 26 mm), and that phone never tracked.

The camera intrinsics come from the main camera's 35 mm-equivalent focal
length. A saved calibration's focal length gives way to a newer profile
unless it was set by hand. getUserMedia streams span the sensor width, so
`f_px = f35 / 43.27 x 1.25 x long side`. The camera-IMU rotation follows
from the screen orientation (W3C device frame).

Two values are measured on the phone instead:

- **Accelerometer sign:** checked against the device orientation. iOS
  reports gravity instead of specific force (the opposite sign).
- **Camera-IMU time offset:** estimated online (`src/sync.rs`). Until a
  phone has been measured, the page assumes −25 ms (a saved offset that was
  only the earlier 0 ms default gives way to it). Once the sync has seen
  enough rotation, the page shifts the frame timestamps, saves the
  offset for that phone, and restarts tracking. On ADVIO it recovers
  injected ±60 ms errors to within 3–6 ms (correlation 0.96–0.99). That
  matters: an offset 30 ms off doubles the error, and 60 ms off resets
  tracking 8–30 times per walk. The page only corrects by 15 ms or more:
  on an iPhone, solid estimates read ±8 ms off depending on the motion,
  and each correction restarts tracking.

The gyroscope's axis order depends on the browser engine. The W3C spec has
`rotationRate.alpha / beta / gamma` about the device's z / x / y axes, as
Chrome on Android reports them. WebKit (Safari, and every browser on iOS)
reports them about x / y / z, and so does Chrome 155 on Android (on a
Nothing Phone (4a) Pro the camera check switched to it within 12 s;
Firefox on Android keeps the spec's order). An iPhone 17-class phone on
iOS 26 showed it: with the spec's order, the camera-IMU rotation measured from the video
was 113–124° off the true one (a 120° turn about the (1, −1, 1) diagonal,
which is what this permutation does). The default follows the platform. If
a solid sync estimate finds the rotation more than 45° off and the other
order explains it to within 30°, the page switches order and saves it for
the phone.

The same estimate also reports a focal scale and the camera-IMU rotation,
but both assume pure rotation, and a hand-held phone always translates a
little. On iPhone recordings they read 7–13% and 7–18° off (on the
Nothing phone the focal scale read 0.92 against the generic 26 mm: its
main camera is 24 mm), where a
bundle adjustment with the gyroscope's rotations (camera positions and
point depths free) found the 24 mm profile right within 2% and the
rotation within 1°. They catch gross errors only (a wrong lens, swapped
axes). Settings are stored per phone model.

Measured on three iPhone recordings with that bundle adjustment (validated
on synthetic data and on ADVIO, where it recovered the known calibration):
focal 24.0 ± 0.4 mm, time offset within 5 ms of the saved one, camera-IMU
rotation within 1°, rolling-shutter readout 3–4 ms, lens distortion and
principal point offset negligible, accelerometer and gyroscope axes
consistent (the best of all 48 signed axis permutations), gyroscope scale
within 0.6%, frames at 29.99 fps with about 1 ms of jitter, and motion
samples at 60.00 Hz with no gaps.

## Phone defaults for the estimator

The estimator starts from Basalt's EuRoC configuration (a stereo rig on a
drone), with these changes for a hand-held phone (`src/vio.rs`, and the
page's `IMU_DEFAULT` in `web/profiles.js`).

- **Feature grid:** 13 cells across the frame's short side (28 px at
  360 px) instead of 50 px. The frontend keeps at most one new point per
  cell, so 50 px capped a phone's frames at about 60 points.
- **Triangulation baseline:** 1 cm instead of 5 cm
  (`vio_min_triangulation_dist`). A phone that mostly turns rarely moves
  5 cm between keyframes, so few tracks became landmarks.
- **Keyframes:** one allowed every 2 frames instead of 5
  (`vio_min_frames_after_kf`), with 2 non-keyframe states in the window
  instead of 3 (`vio_max_states`). A monocular window only gets new
  landmarks at keyframes, and a phone that turns loses most of its tracks
  within 5 frames. The smaller window pays for the extra keyframes.
- **Bias random walks:** 30x stiffer than the white noise's ADVIO x3
  scaling (accelerometer 2.1e-5 m/s²/√s, gyroscope 5.1e-6 rad/s/√s). A
  looser bias lets the window absorb disagreements between the camera and
  the gyroscope (blur, fast pans) into the gyroscope bias. On an iPhone
  that bias ran to 15 mrad/s, where the bundle adjustment above measured at
  most 2.6 mrad/s (iOS corrects the gyroscope's bias itself).
- **First attitude:** roll and pitch come from the specific force averaged
  over the last 0.5 s of motion samples, each rotated to the first frame by
  the gyroscope. Basalt takes the one sample after the first frame, which
  suits a still drone; a hand-held phone accelerates by 0.5–1 m/s², a tilt
  error of 3–6°. The page hands the session the motion samples it already
  has when tracking starts.

On a 15 s iPhone recording (360 x 480, no ground truth):

| Grid, baseline | Points tracked | Landmarks in view | Frames with < 10 | Frames tracking |
| --- | ---: | ---: | ---: | ---: |
| 50 px, 5 cm (EuRoC) | 60 | 13 | 44% | 82% |
| 28 px, 5 cm | 231 | 31 | 33% | 85% |
| 28 px, 1 cm | 231 | 84 | 13% | 95% |

Both cost time: about 2.5x the EuRoC settings' time per frame (13.7 ms
instead of 5.4 ms natively on one thread, on that recording).

The keyframe and bias changes, on three iPhone recordings (each replayed
at five frame time offsets from −10 to +10 ms, 15 runs) and ADVIO's four
walks:

| Keyframes, bias walk | Phone resets | Landmarks in view | Frames tracking | ADVIO ATE | ADVIO scale error | ADVIO resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 frames, 3 states; ADVIO x3 | 9 | 81 | 89% | 3.9 m | 12% | 1 |
| 5 frames, 3 states; 30x stiffer | 3 | 85 | 90% | 3.8 m | 13% | 2 |
| 2 frames, 2 states; 30x stiffer | 4 | 93 | 94% | 2.6 m | 4% | 2 |

All the phone resets are on one recording of fast sweeps (up to 2.6 rad/s).
Either keyframe value alone did worse than both: 2 states alone cut the
landmarks in view to 71 and left ADVIO at 3.9 m; 2 frames alone cost 13%
more time per frame and left ADVIO at 3.2 m. These came out of about 100
configurations on these seven sequences, so treat them as tuned on them:
advio-01 alone moves the mean ATE by more than 0.4 m on small changes.

### Still start

Basalt starts a window at rest: zero velocity, and roll and pitch from the
specific force. Starting each recording every 0.5 s and tracking 3 s (60
starts over four iPhone recordings, the phone moving throughout as the page
used to ask), 27% of the starts reset, nearly all on speed within 0.5–2.5 s.
The window got no landmarks for its first second or so, integrated the
wrong velocity, triangulated its first landmarks against the drifted poses,
and diverged. By the whole recording's estimate of the speed at the start
(on the two recordings that replay in full without a reset), one start in
eight under 0.2 m/s reset, and five in seven above 0.5 m/s. No setting changed that rate (23–28% for the triangulation
baseline, keyframes, window and frontend thresholds).

So the session waits for the phone to be held still before each start and
restart (`StillStart` in `src/vio.rs`; ⚙ → "Wait for the phone to be held
still"), and the page says "Hold still". The motion sensors cannot tell a
steady pan from rest, so it judges from the tracked points' motion left
after the gyroscope's rotation, which only moving through space produces:
its median was 0.5–0.7 px a frame below 0.05 m/s, 1 px at 0.1–0.2 m/s and
2.5 px at 0.3–0.5 m/s. It judges that motion over at least 25 ms (every
frame at 30 Hz, every other at 60). After 0.18 s under 1.2 px (six frames
at 30 Hz) it restarts the estimator with its attitude from those motion
samples alone. After 2 s without stillness it starts anyway, as before.

The recordings cannot show what this does for someone who holds still when
asked: their phones never stop. Replaying them in full (four recordings at
five time offsets), it cut the resets from 15 to 9, and the frames tracking
from 91% to 75% (the waits count as not tracking); ADVIO stayed within its
runs' spread (2.97 to 2.65 m).

### What drops tracks

`replay` reports why the frontend dropped tracks. On the iPhone recordings
the forward-backward check (`optical_flow_max_recovered_dist2`: a point
tracked back must land within 0.2 px of where it started) drops 10–27% of
the tracks each frame, against 5% on ADVIO, and more the faster the phone
turns. Even near still, 15% of the points are lost from one frame to the
next (4% on ADVIO). A 0.3 px check kept a third more landmarks and cut the
phone resets from 15 to 9, but doubled ADVIO's scale error (7% to 12%), so
it stays at 0.2 px.

### Fast motion

When the camera loses all its landmarks (blur, or the points leave the
view in a fast turn), the estimator keeps integrating the IMU. After
landmarks have been seen, a frame with none connected (two in a row) is
reported as `"coast"`, and the page shows "Coasting on IMU" and keeps
drawing the AR view (it used to say "Initializing" and stop it). The
watchdog still resets after 2 s of them while 20 points are tracked, or
when the speed passes 3 m/s (and re-anchors to the pose 1 s back). The
session's timings are in seconds, so they hold at 60 Hz too. In a 70 s phone session with the still
start there were no resets at all; the three dropouts the page showed as
"Initializing" were coasting, the longest for 14 frames (half a second).
On the recordings 0–2% of the frames coast, at most 3 in a row.

Seeding each point's search with the gyroscope's rotation
(`config.optical_flow_imu_seed_rotation`, and a version that integrates
the whole frame interval and seeds the backward search too) does not
help: it removes the drops of points searched out of the image (2% to
0.1% a frame) but not the forward-backward ones, and the resets and ADVIO
did not change. The
points are lost to blur and to leaving the view, not to moving too far
for the pyramid.

iOS offers no exposure control to the page (the camera's capabilities list
frame rate, zoom, focus distance, torch and white balance only). A frame's
exposure cannot outlast the frame, so ⚙ → "Film at 60 fps" (on by
default) halves the longest exposure, and with it the blur of a fast move,
at the cost of more noise. One A/B on an iPhone, two recordings with
similar motion (median 0.25 and 0.23 rad/s), tracking every other frame:
at 60 fps the forward-backward check dropped 5.9% of the points a frame
against 19.2% at 30 fps, leaving 187 landmarks in view against 73, and 2%
of the frames coasting against 16%. A second pair agreed (14.8% against
27.4%). ⚙ → "…and track every frame at 60 Hz" (also on by default) tracks
them all; see "60 Hz".

## 60 Hz

Tracking every frame of a 60 fps camera first made things worse: on
ADVIO (filmed at 60 fps) the mean ATE went from 2.65 m to 7.0 m with 13
resets against 2, and a phone's 60 fps recording reset three times where
the same frames at 30 Hz did not. Neither spacing keyframes in time nor
larger visual noise explained it. Starving 30 fps of IMU samples did: with
a 30 Hz IMU, one sample per frame interval, advio-16 went from 2.0 m to
18.3 m and 9 resets; with a 240 Hz IMU, 60 fps recovered most of the way
(advio-16 3.6 m against 7.1, advio-02 4.0 m against 9.4).

The cause is Basalt's IMU noise model. It injects the IMU's per-sample
noise (density² × rate) into every preintegration step and scales it by
dt², which is right only for steps one sample period long. The steps at
the start and end of each frame interval are shorter, so an interval's
covariance falls short of the continuous model: by about 17% with two
samples per interval (a 60 Hz IMU at 30 fps), and by about a third with
one (60 Hz at 60 fps), swinging with where the samples land. The IMU then
claims more certainty than it has, more so at 60 Hz. Upstream rarely sees
it: its 200 Hz IMU against 20 Hz cameras gives ten steps an interval.
`config.vio_imu_noise_per_step` (on in the phone defaults, off in the
crate) keeps the densities continuous and gives each step variance for
its own length, density² × dt; a test in `pipelines/basalt` checks that an
interval's covariance no longer depends on where its samples fall.

| ADVIO (mean ATE, resets) | 30 Hz | 60 Hz | 60 Hz, estimating every other frame |
| --- | ---: | ---: | ---: |
| pinned noise | 2.65 m, 2 | 6.97 m, 13 | 3.16 m, 4 |
| per-step noise | 2.59 m, 3 | 4.29 m, 2 | 3.12 m, 2 |

At 30 Hz it is neutral (on six phone recordings at five offsets: 15
resets against 17, 67 landmarks in view against 62). On the phone's 60 fps
recordings it is what makes 60 Hz work: no resets at three offsets (three
before), 80–89% of frames tracking against 75–84% for the same frames at
30 Hz, and on a second recording 172 landmarks in view against 94.
ADVIO's 60 fps frames carry the 30 fps frames' exposure, so it shows only
the costs of 60 Hz (more frames for the forward-backward check to fail;
its walks stay 1.7 m behind 30 Hz), not the halved blur.

`Session::estimate_every` (`--estimate-every 2`, `vio.setEstimateEvery`)
tracks every frame but runs the estimator on every other one and carries
the pose to the rest with the IMU. It costs about a quarter less, but on the
phone it reset once in three offsets and its carried poses jump at each
estimate, so 60 Hz estimates every frame.

## The GPU wait and the estimator, overlapped

With the frontend on the GPU, a frame's GPU round trip (about 4 ms on an
iPhone) and the estimator would add up. The worker runs the frontend one
frame ahead instead: when a frame's tracks are back, the next waiting
frame goes to the GPU before the estimator runs on the first
(`vio.submit`, `vio.takeGpu`, `vio.estimateGpu`). The next frame's
tracking depends only on the frame before, so this gives the same poses
(`examples/gpu_check.rs` compares a pipelined and a frame-by-frame session:
identical on 300 frames). A restart cancels the frame on the GPU, which is
dropped. With a fake session in Node at 60 Hz (4 ms GPU): with a 14 ms
estimator it kept 119 of 120 frames (one after another, about 111), at 20
ms 97 (the estimator alone allows 100), at 30 ms 66 (67).

## The frontend on the GPU

Basalt's frontend (the pyramid, the frame-to-frame KLT with its
forward-backward check, and the grid FAST) also runs on the GPU, with wgpu:
WebGPU in the browser, Vulkan or Metal natively (`src/gpu_flow.rs`,
`src/flow.wgsl`). It is on where the browser has WebGPU (⚙ → "Track points
on the GPU"); the estimator stays on the CPU. A frame is one submission
(the pyramid; one workgroup per tracked point, a thread per patch sample,
forward then back; the grid cells that still hold a point; one workgroup
per empty cell for FAST) and one readback. In the worker that is
`vio.submit(t, frame)`, await the promise, then `vio.takeGpu()` and
`vio.estimateGpu()` (see "The GPU wait and the estimator, overlapped");
natively `replay --gpu` and `advio --gpu`.

Against the CPU frontend, each frame starting from the CPU's tracks
(`examples/gpu_check.rs`, 120 frames of an iPhone recording): of 28,356
tracked points, both kept 22,648 and both lost 5,704 (5,703 for the same
reason), and the GPU alone kept 4. Kept points differ by 0.00001 px
(median), 0.002 px at most. 5,859 of 5,873 new corners are the same pixel;
the others are ties. The arithmetic is plain f32, where the CPU code
reproduces Eigen's rounding order, so it agrees to rounding, not bit for
bit. End to end (six iPhone recordings at five time offsets, and ADVIO's
four walks), the GPU frontend left 19 resets on the phones against 17 and
60 landmarks in view against 62; on ADVIO 2.65 m mean ATE both (4.03, 2.68,
1.85 and 2.02 m against 4.07, 2.65, 1.85 and 2.01), 3 resets against 2:
the spread any small change shows.

On one native thread the frontend was 7.4 ms of a 16 ms frame. On the GPU
a frame costs 0.3 ms to submit and about 1 ms of waiting. In Chrome the
wait was 4–8 ms: WebGPU hands a buffer back through the GPU process and a
promise, however little the work. This machine was too loaded during the
browser tests to compare frame times there; the page reports the split
(`gpuSubmitAvg`, `gpuWaitAvg`, `finishAvg` in the telemetry). With the
frontend off the CPU, the wasm threads only serve the estimator, so the
single-threaded build (⚙ → "Use wasm threads" off, no thread pool at all)
is worth comparing on a phone.

Browsers compile WGSL more strictly than wgpu does natively. Chrome
rejected a branch on a workgroup variable read outside
`workgroupUniformLoad` (barriers must stay in uniform control flow), a
`let` declared twice in one scope, and an f32 literal at the edge of the
range; all three ran natively. A device error now goes to the page's log,
and the session falls back to the CPU frontend.

Not on the GPU yet: the main thread still draws each frame to a canvas,
reads it back and greys it (2.4–5 ms a frame on an iPhone, `grabMsAvg`).
WebGPU could take the video frame directly in the worker.

## Firefox and slower phones

Firefox on Android has no WebGPU, so it tracks on the CPU frontend. It
was slow there: on an 8-core Android phone with wasm threads (4 workers),
a frame took 13 ms before the first landmarks but 68–120 ms once
tracking, 7–10 fps. The estimator was the cost, not the frontend. Chrome on a Nothing
Phone (4a) Pro, a mid-range phone, took 17–45 ms with its frontend on the
GPU. Desktop Firefox was no guide: Playwright's Firefox build ran wasm on
its baseline compiler only (a float loop 14x slower than in Chromium, while
its JavaScript was faster), so neither the SIMD build nor other
optimization levels could be judged there. The phone's telemetry is the
measure.

⚙ → "Lighter tracking for slower phones" (Auto: on in Firefox) trades
accuracy for speed. It sets a coarser feature grid (a tenth of the frame's
short side: 36 px at 360 px, against 28), at most 3 solver iterations
(against 7), and the estimator on every other frame (`estimate_every` 2:
the IMU carries the pose to the frames between). Natively, on six iPhone
recordings at five time offsets (30 Hz) and ADVIO's four walks:

| | phone resets | landmarks in view | ms a frame | ADVIO mean ATE | resets | ms a frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phone defaults | 15 | 67 | 9.3 | 2.59 m | 3 | 19.1 |
| grid 36 px | 20 | 39 | 6.2 | 2.69 m | 3 | 14.4 |
| grid 36 px, 3 iterations | 16 | 41 | 4.9 | 4.21 m | 5 | 11.8 |
| estimating every other frame | 14 | 51 | 5.8 | 4.23 m | 1 | 13.0 |
| grid 36 px, every other frame | 13 | 36 | 5.3 | 3.37 m | 0 | 13.6 |
| **light: grid 36 px, 3 iterations, every other frame** | 10 | 35 | 3.8 | 3.72 m | 4 | 8.8 |

The light setting costs 2.4x less a frame on the phone recordings (2.2x on
ADVIO), with fewer resets but fewer landmarks, and its walks end further
off (scale error 11% against 5%). The timings are from a shared, loaded
machine, so they are rough. Where the phone cannot keep up, frames are
dropped and the KLT has to bridge larger motions, so speed matters more
there than the solver's last iterations. If Firefox's 68–120 ms falls in
proportion, that is about 30–50 ms a frame: 20–30 fps while tracking, not
yet a steady 30. Firefox reports no phone model, so the page uses the
default 26 mm profile there.

## Results on ADVIO (iPhone 6s, browser-like input)

Full sequences, IMU white noise at 3x ADVIO's Kalibr values and bias
walks 10x below them (the phone defaults' noise), with the watchdog, the
phone defaults above, and each sequence's measured frame time offset
(`--td-ms`: -300, -300, -312, -328 ms). ATE is SE(3)-aligned to ADVIO's
ground truth. ARKit's own estimate over the same frames is shown for
reference. In parentheses: the EuRoC file's grid, baseline and keyframe
values, with the bias walks at 3x Kalibr like the white noise.

| Sequence | Path | Resets | Frames tracking | ATE | Scale | ARKit ATE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advio-01 | 244 m | 1 (6) | 99% (97%) | 4.1 m (5.6 m) | 0.98 (0.76) | 2.5 m |
| advio-02 | 221 m | 0 (0) | 99% (92%) | 2.7 m (6.8 m) | 1.01 (1.14) | 2.6 m |
| advio-13 | 142 m | 1 (1) | 99% (98%) | 1.9 m (2.2 m) | 1.05 (1.07) | 1.3 m |
| advio-16 | 151 m | 0 (1) | 98% (94%) | 2.0 m (4.0 m) | 1.07 (1.04) | 1.2 m |

The mean ATE drops from 4.7 to 2.6 m, and advio-02 matches ARKit. Runs
are deterministic for a given thread count: the table is single-threaded.
advio-01 is chaotic: without the still start it ends at 5.4 m, or 3.8 m
with rayon's default threads, while the other three move by under 0.1 m.

Without the watchdog, three of the four diverge for good within 150 s. The
triggers were a near-static, low-parallax start, or a stretch with few
landmarks. After that, a runaway gyroscope or accelerometer bias absorbs
the visual residuals and the speed grows without bound. ADVIO's frame
timestamps run about 300 ms behind its IMU clock (`--td-ms`).

```bash
cargo run --release --example advio -- ~/datasets/advio/advio-02 --td-ms -300 --noise-scale 3 --bias-scale 0.1
```

Time per frame with the phone defaults, measured before upstream's faster
KLT frontend (#354):

| Build | ADVIO, 360 x 640 | iPhone recording, 360 x 480 |
| --- | ---: | ---: |
| Native, 1 thread (Ryzen 9 9950X) | 14–17 ms | 13–15 ms |
| wasm in Node, 1 thread | 37 ms | 26 ms |
| Chrome, 4 wasm threads | ~22 ms | |
| iPhone 16 Pro / 17 class, 3 wasm threads (28 px, 5 cm) | | 18 ms median |

#354 takes 20–26% off a frame in wasm (Node, three iPhone recordings:
24.2 → 18.0, 29.7 → 22.4 and 19.0 → 15.2 ms) and about 19% natively, with
the same poses to the last digit.

The Chrome figure is for the demo, which tracks every frame of its 30 fps
video. `web/node_replay.mjs` checks the wasm build against the native
replay: they end 1 mm apart after 4.6 m of advio-13, and 0.3 m apart after
12.5 m of the demo. They are not bit-identical because wasm uses Rust's
libm, and the difference grows with the number of landmarks.

Where the time went before #354, natively on one thread on two iPhone
recordings: the frontend took 7.4 ms a frame whatever the scene (temporal
KLT 3–5 ms, FAST replenishment 2.4–4.5 ms, the pyramid 0.1 ms), and the
estimator 3–9 ms, growing with the landmarks in view (most of it the per-landmark
reduction inside its 7 Levenberg-Marquardt iterations; up to 21 ms on a
keyframe with marginalization). On the phone a frame took 7–42 ms over a
70 s session, 18 ms typically, more with more landmarks; in its last
third frames were skipped. Three threads instead of one are 1.7x faster
natively.

For tracking at 60 fps:

- **Exact FAST rejection** from the four compass pixels took 9–18% off
  natively and 11% in wasm; upstream's faster frontend (#354) does the same
  and more.
- **Fewer iterations:** `vio_max_iterations` 3 instead of 7 takes 27% off
  and left 4 resets on the phone recordings instead of 9, but advio-02
  went from 2.65 m to 5.3 m (mean 3.4 m), so it stays at 7.
- **Not every frame:** on ADVIO, which is filmed at 60 fps, tracking every
  frame gave a mean ATE of 7.0 m and 13 resets against 2.65 m and 2 at
  30 fps; with the keyframe spacing and the window doubled to span the
  same time, 6.4 m and 6 resets. The cause was the IMU noise model (see
  "60 Hz"); with `vio_imu_noise_per_step` the page films and tracks every
  frame at 60 fps by default.
- **WebGPU** suits the frontend (pyramid, KLT and FAST are per-pixel and
  per-point work), not the estimator (small dense solves in a sequential
  loop). The frontend now runs there; see "The frontend on the GPU".

Five changes in `pipelines/basalt`. All are identical natively with the
pinned configuration, and the crate's tests pass unchanged.

- **Hash buckets:** the 64-bit libstdc++ bucket arithmetic is now width
  independent.
- **No environment:** wasm32 has no environment or process id, so the
  diagnostic probes skip them.
- **`fused_mul_add` (`src/fma.rs`):** WebAssembly has no scalar FMA, so
  each `f32::mul_add` was an out-of-line software `fmaf`, 40% of a frame.
  On wasm it is now an inline f64 multiply-add.
- **`vio_min_triangulation_dist`:** the VIO's landmark baseline gate was
  hard-coded at 5 cm. It now reads the configuration value, as upstream
  does (the pinned EuRoC file says 5 cm).
- **`vio_obs_std_dev`, `vio_obs_huber_thresh`:** the visual residuals'
  0.5 px standard deviation and 1 px Huber threshold were hard-coded. They
  now read the configuration (the pinned EuRoC file has the same values).

## The overlay, matched to the image

The tracker finishes a frame a frame or two after it is on screen, and
the video shows camera frames as they come, ahead of anything the page
draws: on an iPhone a frame reaches the page about 14 ms after capture,
and WebKit's frame callbacks give no display time (their
`expectedDisplayTime` is the capture time). So each overlay is drawn for
the image that will be on screen when it appears: captured one display
refresh after the draw, less the camera's latency to the page (both
measured as the page runs; ⚙ → "Overlay timing" shifts it by eye). The
pose is the last estimate carried to that time by the gyroscope
(bias-corrected; past the newest motion sample, its rate held) and by the
velocity, with the camera's offset from the IMU turned by the new
attitude. Drawing for the newest frame the page had seen left the overlay
a frame behind the video. The telemetry reports the latency, the refresh
and how far the overlay runs past the newest motion sample and estimate
(`overlay`). The tracked points come
from the older frame, so they are hidden by default; ⚙ → "Show tracked
points" draws them turned onto the frame on screen by the rotation in
between.

## Telemetry

The page reports what it sees to the server hosting it (`web/telemetry.js`,
`POST /telemetry`). `serve.py` appends each page load's events to
`telemetry/<session>.jsonl` and prints the important ones to its log. No
images are sent. The session id is shown on the start sheet, and
`?telemetry=0` turns reporting off. `?debug=1` shows the same snapshot on
screen.

- **Device:** the browser, phone model, cores and cross-origin isolation,
  whether it has WebGPU (and the adapter), the camera list and the chosen
  camera's settings, and whether points are tracked on the GPU.
- **Clocks:** the first motion event and video frames with their raw
  timestamps, and which clock each one ended up on.
- **Flow:** every 2 s, counters of frames grabbed, skipped, posted,
  processed and dropped. Also motion events received and accepted, how far
  the frames run ahead of the IMU, and the worker's own state (pending frame
  vs. newest IMU sample, last frame, queue). Also a recent frame's
  brightness and contrast (`frameLevel`: a blank or dark view tracks no
  points, which is not a tracking failure), the main thread's time to
  read a frame back, and on the GPU frontend the time to submit a frame,
  to wait for its points and to finish it.
- **Tracking:** status changes and watchdog resets (with the reason), and
  the time-sync estimate. That includes the measured camera-IMU rotation
  against the rig's.
- **Failures:** errors from the page and the worker (including wasm panics),
  and stalls: no result for 3 s, with the likely cause. A threaded worker
  that stops answering is replaced by the single-threaded build.

This found the first phone bug. WebKit (Safari, and Chrome on iOS) reports
`mediaTime` 0 for every camera frame, and the page skipped frames whose
`mediaTime` repeated, so it tracked only the first one. Frames are now
told apart by `presentedFrames`.

## IMU only

The **IMU only** button ignores the camera. The map (landmarks, cube)
freezes, and the camera pose comes from integrating the motion samples
from the last tracked state: the gyroscope for the attitude, and the
accelerometer less gravity for velocity and position, both with the
estimated biases removed. If the frozen points stay on their scene
features while the phone turns, the gyroscope axes, the camera-IMU
rotation, the focal length and the time offset all agree. Position drifts
within seconds, as a double-integrated phone accelerometer does (about
10 m in 5 s on the walking demo). The VIO keeps tracking in the
background. The status line shows how far the integrated pose has drifted
from it, and the telemetry logs that drift every second
(`imu-only-drift`).

## Recording and replaying a phone session

⚙ → Record captures what the page feeds the VIO: grey frames (every
camera frame, also those not tracked when filming at 60 fps), motion
samples, rig, device details and calibration, up to about 210 MB (40 s at
30 fps, 20 s at 60). The recording can be
downloaded, or uploaded to the server hosting the page (`recordings/`). The
upload goes in 4 MiB chunks. A failed chunk is retried, and a retry resumes
where the server's copy ends. The file stores the motion samples before
the frames, so a recording cut short still replays up to its last whole
frame. Replay it natively:

```bash
cargo run --release --example replay -- recordings/<id>.vrec [--td-ms 20] [--fx 300] [--noise-scale 2] \
    [--set config.vio_min_triangulation_dist=0.02]
```

Besides resets and timing, the replay reports how many landmarks stayed in
view, how often tracking held and why the frontend dropped tracks: with no
ground truth, those are what to tune a phone recording by. `--skip-s` and
`--max-s` replay a slice (a fresh start at any point),
`--no-still-start` starts without waiting for a still phone, `--gpu`
tracks on the GPU, `--every 2` replays every other frame (a 60 fps
recording at 30 Hz) and `--estimate-every 2` runs the estimator on every
other frame.

## Limits

- **Experimental:** monocular-inertial is the port's experimental mode, and
  it is not upstream-tuned for phones.
- **Hold still, then move:** the estimator starts at rest, then needs
  sideways motion to triangulate landmarks and observe scale. Waving the
  phone through the start is the commonest failure (see "Still start").
- **No relocalization:** there is no loop closure. After a watchdog reset
  the trajectory continues from the last good pose, but drift is not
  corrected.
- **Browser IMU:** motion events arrive at about 60 Hz with jittery
  timestamps, one or two per frame (see "60 Hz"). Rolling shutter is not
  modelled.
- **No depth:** iPhone browsers expose neither ARKit's tracking nor the
  LiDAR or TrueDepth depth (no WebXR AR on iOS; the camera gives colour
  only). Android Chrome's WebXR does expose ARCore's pose and depth.
- **Phone profiles are approximate:** they are not per-unit calibrations.
  Lens distortion is assumed corrected by the phone's video pipeline.
