// Live monocular-inertial VIO on a phone: camera frames and motion events
// in, the wasm session (worker.js) tracks, the page draws the camera with an
// AR overlay and a map of the trajectory.
import * as THREE from "three";
import { OrbitControls } from "three";
import { detectDevice, loadCalibration, saveCalibration, clearCalibration, defaultCalibration, intrinsics, IMU_DEFAULT } from "./profiles.js";
import { recordingBlob } from "./recording.js";
import { emit, flush, sessionId, environment, enabled as telemetryOn, hasServer, text } from "./telemetry.js";

const $ = (id) => document.getElementById(id);
const DEG = Math.PI / 180;
// ?raw=1: draw the AR overlay at the tracked frame's own pose (no gyro
// prediction), so landmarks must land exactly on their tracked points.
const RAW_POSE = new URLSearchParams(location.search).has("raw");
// ?debug=1: the diagnostics snapshot sent to the server, live on screen.
const DEBUG = new URLSearchParams(location.search).has("debug");

// ---------------------------------------------------------------- state

let device = null;
let calib = null;
const prefs = {
  width: Number(localStorage.getItem("visloc-vio-width") || 360),
  watchdog: localStorage.getItem("visloc-vio-watchdog") !== "0",
  stillStart: localStorage.getItem("visloc-vio-still") !== "0",
  // 60 fps, every frame tracked: on unless turned off. New keys: the
  // settings sheet saved every option under the old off default.
  fps60: localStorage.getItem("visloc-vio-60fps:v2") !== "0",
  track60: localStorage.getItem("visloc-vio-track60:v2") !== "0",
  tracks: localStorage.getItem("visloc-vio-tracks") === "1",
  overlayLeadMs: Number(localStorage.getItem("visloc-vio-overlay-lead") || 0),
  light: localStorage.getItem("visloc-vio-light") || "auto", // "auto" | "on" | "off"
  gpu: localStorage.getItem("visloc-vio-gpu") !== "0",
  threads: localStorage.getItem("visloc-vio-threads") !== "0",
  landmarks: localStorage.getItem("visloc-vio-landmarks") !== "0",
};

let worker = null;
let workerReady = null; // { threaded, threads, webgpu }
let frontendGpu = null; // the GPU tracking points (its description), or null
let workerCreatedAt = 0;
let running = false;
let serverHere = true; // serve.py, with telemetry and uploads (see hasServer)
let stream = null;
let rig = null;
let accelSign = 1;
let screenAngle = 0;
let frameW = 0, frameH = 0;
let inflight = 0;
let videoFrameCallback = 0, frameRequest = 0;
// Demo mode: a recorded video and its IMU samples instead of the sensors.
let demo = null; // { fps, frameTimes (s), imu: [[t, gx, gy, gz, ax, ay, az]], rig, next }

const capture = document.createElement("canvas");
const captureCtx = capture.getContext("2d", { willReadFrequently: true });

// IMU: samples waiting to be sent, a short history (AR prediction), rate.
let imuBatch = [];
const imuHistory = []; // { t (ms), g: [x, y, z] rad/s, a: [x, y, z] m/s² specific force }
let imuCount = 0, imuRateHz = 0, imuRateWindow = performance.now();
let lastMotion = null; // { t, raw } for timestamp smoothing
let motionPeriod = 0; // ms between motion events, learned from their times
// "event": event.timeStamp (performance.now() ms, per the spec); "receipt":
// performance.now() when it arrived, where timeStamp is on another clock.
let motionClock = null;
let lastAccel = [0, 0, 9.81];

// Latest results.
let latest = null;
let lastFrameTime = 0; // ms, of the newest grabbed frame
// For drawing the overlay at the image on screen: how long a camera frame
// takes from capture to the page (frame callback), and the display's
// refresh interval (the overlay shows about one refresh after it is drawn).
let videoLatencyMs = 15, refreshMs = 16.7, lastRafAt = 0;
let lastPostedTime = -Infinity; // ms, of the newest frame sent to the tracker
const trail = new Float32Array(3 * 200000);
let trailCount = 0, distance = 0, lastTrailPos = null;
let results = 0, resultsWindow = performance.now(), fps = 0, msAvg = 0;
let grabMsAvg = 0; // main thread: drawing a frame, reading it back and greying it
// Brightness and contrast of a recent frame (mean and standard deviation of
// the grey levels), to tell a blank or dark view from a tracking failure.
let frameLevel = null;

/** Mean and standard deviation of every 16th grey pixel. */
function greyLevel(gray) {
  let n = 0, sum = 0, sum2 = 0;
  for (let i = 0; i < gray.length; i += 16) {
    sum += gray[i];
    sum2 += gray[i] * gray[i];
    n++;
  }
  const mean = sum / n;
  return { mean: Math.round(mean), std: Math.round(Math.sqrt(Math.max(0, sum2 / n - mean * mean))) };
}
let gpuWaitAvg = 0; // worker: submitting a frame to the GPU until its points are back
let gpuSubmitAvg = 0, finishAvg = 0; // worker: before the wait, and the estimator after it
let cubePlaced = false;
const trackAge = new Map();
// Online camera-IMU synchronization (see src/sync.rs).
let lastSync = null;
let syncApplied = 0;
const recentSync = []; // offsets of the latest estimates, ms

// Diagnostics (telemetry.js): cumulative counters of the frame and motion
// flow, the worker's latest stats and when they came, the current stall.
const diag = {
  callbacks: 0, // video frame callbacks
  duplicates: 0, // callbacks that brought no new frame
  skippedBusy: 0, // frames not grabbed: two were in flight
  skippedRate: 0, // frames not grabbed: 60 fps capture tracks every other one
  posted: 0,
  results: 0,
  sessionResults: 0,
  dropped: 0,
  errors: 0,
  motion: 0, // devicemotion events
  motionIncomplete: 0, // events without acceleration or rotation rate
  motionSnaps: 0, // smoothed motion times reset to the event times
  imuSent: 0, // samples posted to the worker
  framesLogged: 0,
  lastCallbackAt: 0,
  lastMotionAt: 0,
  lastResultAt: 0,
  sessionAt: 0,
};
let workerStats = null, workerBeatAt = 0;
let stall = null; // { at, reason } while no results arrive
let lastStatus = null;
let restartingWorker = false;

// Recording.
let recording = null; // { frames: [], times: [], imu: [], started }
let lastRecording = null;
// A recording keeps every camera frame (60 a second when filming at 60
// fps, tracked or not), up to about 210 MB: 40 s at 30 fps, 20 s at 60.
const REC_MAX_BYTES = 210e6;

// ---------------------------------------------------------------- helpers

function toast(text, ms = 2200) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), ms);
}

function setStatus(state, text, sub = "") {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
  $("status-sub").textContent = sub;
}

/** The conjugate (inverse) of the unit quaternion `q` = [w, x, y, z]. */
function quatConj(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

function quatMul(a, b) {
  // [w, x, y, z]
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function quatFromRotvec(v) {
  const angle = Math.hypot(v[0], v[1], v[2]);
  if (angle < 1e-12) return [1, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]];
  const s = Math.sin(angle / 2) / angle;
  return [Math.cos(angle / 2), v[0] * s, v[1] * s, v[2] * s];
}

/** T_imu_cam for the rear camera with the screen rotated by `angle`
 * degrees counter-clockwise: image x and y in the device frame, the optical
 * axis out of the back. Returns [qw, qx, qy, qz]. */
function imuFromCameraRotation(angle) {
  const a = angle * DEG;
  const c = Math.cos(a), s = Math.sin(a);
  const x = [c, -s, 0], y = [-s, -c, 0], z = [0, 0, -1];
  const m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
  const q = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().set(m[0][0], m[0][1], m[0][2], 0, m[1][0], m[1][1], m[1][2], 0, m[2][0], m[2][1], m[2][2], 0, 0, 0, 0, 1),
  );
  return [q.w, q.x, q.y, q.z];
}

function currentScreenAngle() {
  const a = screen.orientation?.angle ?? window.orientation ?? 0;
  return ((Math.round(a / 90) * 90) % 360 + 360) % 360;
}

// ---------------------------------------------------------------- start screen

function describeDevice(el) {
  const rows = [
    ["phone", device.label + (device.matched ? "" : "  (no profile: generic)")],
    ["focal", `${calib.focal35} mm (35 mm eq.)`],
    ["time offset", `${calib.timeOffsetMs} ms`],
    ["IMU noise", `x${calib.noiseScale}`],
    ["build", workerReady ? (workerReady.threaded ? `wasm threads x${workerReady.threads}` : "wasm, 1 thread") : "loading…"],
    ["points", frontendGpu ? `GPU (${frontendGpu})` : "CPU"],
    ["diagnostics", telemetryOn && serverHere ? sessionId : "off"],
  ];
  el.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

async function boot() {
  createWorker(prefs.threads);
  device = await detectDevice();
  calib = loadCalibration(device);
  describeDevice($("device"));
  emit("boot", { ...environment(), device, calib, prefs });
  reportWebGpu();
}

/** Whether this browser offers WebGPU, and what (for planning GPU work). */
async function reportWebGpu() {
  if (!navigator.gpu) return emit("webgpu", { available: false });
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const info = adapter?.info;
    emit("webgpu", {
      available: !!adapter,
      info: info && { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
      features: adapter ? [...adapter.features] : [],
      limits: adapter && {
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
  } catch (err) {
    emit("webgpu", { available: false, error: text(err) });
  }
}

function createWorker(threads) {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => {
    emit("error", { where: "worker", message: e.message || "the worker script failed to load", source: `${e.filename ?? ""}:${e.lineno ?? ""}` });
    setStatus("error", "Error", "the VIO worker failed");
  };
  worker.onmessageerror = () => emit("error", { where: "worker", message: "a worker message could not be read" });
  workerCreatedAt = workerBeatAt = performance.now();
  worker.postMessage({ type: "init", threads });
}

function onWorkerMessage(event) {
  const m = event.data;
  switch (m.type) {
    case "ready":
      workerReady = m;
      emit("worker-ready", { threaded: m.threaded, threads: m.threads, webgpu: m.webgpu, ms: Math.round(performance.now() - workerCreatedAt) });
      if (device) describeDevice($("device"));
      break;
    case "stats":
      workerStats = m;
      workerBeatAt = performance.now();
      break;
    case "log":
      console.log("[worker]", m.message);
      emit("worker-log", { level: m.level ?? "log", message: m.message });
      break;
    case "started":
      frontendGpu = m.gpu ?? null;
      emit("frontend", { gpu: frontendGpu, wanted: prefs.gpu });
      if (device) describeDevice($("device"));
      break;
    case "dropped":
      inflight = Math.max(0, inflight - 1);
      diag.dropped++;
      break;
    case "result":
      inflight = Math.max(0, inflight - 1);
      onResult(m);
      break;
    case "error":
      // Only frames that failed were in flight.
      if (m.frameT != null) inflight = Math.max(0, inflight - 1);
      diag.errors++;
      console.log("[worker error]", m.message);
      emit("error", { where: "worker", message: m.message, during: m.during, frameT: m.frameT, ms: m.ms });
      setStatus("error", "Error", m.message.slice(0, 80));
      break;
  }
}

// ---------------------------------------------------------------- sensors

function onMotion(e) {
  const a = e.accelerationIncludingGravity;
  const r = e.rotationRate;
  const now = performance.now();
  diag.motion++;
  diag.lastMotionAt = now;
  if (!a || !r || a.x == null || r.alpha == null) {
    if (diag.motionIncomplete++ === 0) emit("motion-incomplete", { accel: a?.x != null, rotation: r?.alpha != null, interval: e.interval });
    return;
  }
  // Event times are performance.now() milliseconds; one far from now is on
  // another clock, and then the receipt time is used.
  if (motionClock === null) {
    motionClock = Math.abs(e.timeStamp - now) < 1000 ? "event" : "receipt";
    motionPeriod = e.interval > 0 ? (e.interval < 1 ? e.interval * 1000 : e.interval) : 16.67; // iOS reports seconds
    emit("motion-first", { timeStamp: e.timeStamp, now, interval: e.interval, clock: motionClock, rotationRate: [r.alpha, r.beta, r.gamma], accel: [a.x, a.y, a.z] });
  }
  const raw = motionClock === "event" ? e.timeStamp : now;
  // Smooth the times' jitter towards the event period, learned from the
  // times (the reported interval can be off); back to the event time after
  // a gap or a burst.
  let t = raw;
  if (lastMotion) {
    const gap = raw - lastMotion.raw;
    if (gap > 0 && gap < 200) motionPeriod += 0.02 * (gap - motionPeriod);
    const model = lastMotion.t + motionPeriod;
    t = model + 0.1 * (raw - model);
    if (Math.abs(t - raw) > 40) {
      t = raw;
      diag.motionSnaps++;
    }
    t = Math.max(t, lastMotion.t + 0.25 * motionPeriod);
  }
  lastMotion = { t, raw };
  // rotationRate in deg/s, about the device axes calib.gyroAxes says.
  const g = (GYRO_AXES[calib.gyroAxes] ?? GYRO_AXES.w3c).rows.map((row) => (row[0] * r.alpha + row[1] * r.beta + row[2] * r.gamma) * DEG);
  lastAccel = [a.x, a.y, a.z];
  const s = accelSign;
  const sample = [Math.round(t * 1e6), g[0], g[1], g[2], s * a.x, s * a.y, s * a.z];
  const entry = { t, g, a: [sample[4], sample[5], sample[6]] };
  imuHistory.push(entry);
  if (imuHistory.length > 240) imuHistory.splice(0, imuHistory.length - 240);
  imuCount++;
  if (now - imuRateWindow > 1000) {
    imuRateHz = (1000 * imuCount) / (now - imuRateWindow);
    imuCount = 0;
    imuRateWindow = now;
  }
  if (!running) return;
  if (imuOnly) integrateImu(entry);
  imuBatch.push(...sample);
  if (recording) recording.imu.push(...sample);
  if (imuBatch.length >= 7 * 3) flushImu();
}

function flushImu() {
  if (!imuBatch.length) return;
  const buf = new Float64Array(imuBatch);
  imuBatch = [];
  diag.imuSent += buf.length / 7;
  worker.postMessage({ type: "imu", samples: buf.buffer }, [buf.buffer]);
}

// Device (x, y, z) rates from rotationRate's (alpha, beta, gamma): the W3C
// spec has alpha about z, beta about x, gamma about y; WebKit (Safari and
// every iOS browser) reports them about x, y, z.
const GYRO_AXES = {
  w3c: { label: "W3C (alpha about z)", rows: [[0, 1, 0], [0, 0, 1], [1, 0, 0]] },
  webkit: { label: "iOS (alpha about x)", rows: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] },
};
let gyroAxesSwitches = 0;

const matMul = (a, b) => a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
const transpose = (a) => [0, 1, 2].map((i) => [a[0][i], a[1][i], a[2][i]]);
function quatMatrix([w, x, y, z]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}
const rotationDeg = (m) => Math.acos(Math.max(-1, Math.min(1, (m[0][0] + m[1][1] + m[2][2] - 1) / 2))) / DEG;

/** A solid sync estimate whose camera-IMU rotation is far from the rig's
 * means the gyroscope's axes are mixed up: if the other axis order explains
 * the measurement, switch to it (saved for this phone) and restart. */
function checkGyroAxes(e) {
  if (demo || gyroAxesSwitches >= 2) return false;
  if (!(e.correlation > 0.9 && e.axisSpread >= 0.3 && e.intervals >= 200 && e.rotationErrorDeg > 45)) return false;
  const measured = quatMatrix(Array.from(e.imuFromCamera));
  const assumed = quatMatrix(rig.imuFromCamera.slice(0, 4));
  const current = GYRO_AXES[calib.gyroAxes] ?? GYRO_AXES.w3c;
  for (const [name, axes] of Object.entries(GYRO_AXES)) {
    if (axes === current) continue;
    // The measurement in the other order's device axes, against the rig.
    const predicted = matMul(matMul(axes.rows, transpose(current.rows)), measured);
    const deg = rotationDeg(matMul(transpose(assumed), predicted));
    if (deg > 30) continue;
    emit("gyro-axes", { from: calib.gyroAxes, to: name, residualDeg: Math.round(deg * 10) / 10, rotationErrorDeg: e.rotationErrorDeg, correlation: e.correlation });
    calib.gyroAxes = name;
    saveCalibration(device, calib);
    gyroAxesSwitches++;
    toast(`Gyroscope axes corrected to ${axes.label} (saved for this phone)`, 3500);
    imuBatch = [];
    beginSession("gyro-axes");
    return true;
  }
  return false;
}

let orientation = null;
function onOrientation(e) {
  if (e.beta != null) orientation = { beta: e.beta, gamma: e.gamma };
}

/** +1 if the accelerometer reports specific force (+g up, the W3C
 * convention), -1 if it reports gravity: compares the reading with the up
 * direction implied by the device orientation's beta / gamma. */
function decideAccelSign() {
  if (calib.accelSign === 1 || calib.accelSign === -1) return calib.accelSign;
  const a = lastAccel;
  let up;
  if (orientation) {
    const b = orientation.beta * DEG, g = orientation.gamma * DEG;
    up = [-Math.cos(b) * Math.sin(g), Math.sin(b), Math.cos(b) * Math.cos(g)];
  } else {
    // No orientation events: assume the phone is upright-ish or face up.
    up = Math.abs(a[1]) > Math.abs(a[2]) ? [0, 1, 0] : [0, 0, 1];
  }
  return a[0] * up[0] + a[1] * up[1] + a[2] * up[2] >= 0 ? 1 : -1;
}

async function requestMotionPermission() {
  const asks = [];
  if (typeof DeviceMotionEvent !== "undefined" && DeviceMotionEvent.requestPermission) asks.push(DeviceMotionEvent.requestPermission());
  if (typeof DeviceOrientationEvent !== "undefined" && DeviceOrientationEvent.requestPermission) asks.push(DeviceOrientationEvent.requestPermission());
  const answers = await Promise.all(asks.map((p) => p.catch((err) => String(err))));
  emit("motion-permission", { answers });
  if (answers.some((a) => a !== "granted")) throw new Error("Motion sensor access was denied. Reload and allow it to use VIO.");
}

// ---------------------------------------------------------------- start / stop

async function start() {
  $("start-err").textContent = "";
  $("b-start").disabled = true;
  emit("start");
  try {
    // iOS asks for motion access only from a tap, before anything awaits.
    const permission = requestMotionPermission();
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser has no camera access (getUserMedia). Use https and Safari / Chrome.");
    const cam = navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, ...videoConstraints() } });
    await permission;
    window.addEventListener("devicemotion", onMotion);
    window.addEventListener("deviceorientation", onOrientation);
    stream = await preferMainCamera(await cam);
    const track = stream.getVideoTracks()[0];
    emit("camera", { label: track?.label, settings: track?.getSettings?.() ?? null, capabilities: track?.getCapabilities?.() ?? null });
    const video = $("video");
    video.srcObject = stream;
    await video.play();
    await waitFor(() => video.videoWidth > 0, 4000, "the camera did not start");
    // Some Android phones (a Galaxy Note20 Ultra: 2 events in the first 3 s)
    // take seconds to start delivering motion events.
    if (imuHistory.length <= 10) $("start-err").textContent = "Waiting for the motion sensors…";
    await waitFor(() => imuHistory.length > 10, 10000, "no motion events arrived (is this a phone? are motion sensors allowed in the site settings?)");
    $("start-err").textContent = "";
    if (!workerReady) await waitFor(() => workerReady, 15000, "the WebAssembly module did not load");
    $("start-sheet").hidden = true;
    $("controls").hidden = false;
    $("metrics").hidden = false;
    $("map-wrap").hidden = !mapVisible;
    resizeAll();
    beginSession("start");
  } catch (err) {
    console.log(err);
    emit("start-failed", { name: err?.name, message: text(err), motionEvents: diag.motion, video: [$("video").videoWidth, $("video").videoHeight], workerReady: !!workerReady });
    $("start-err").textContent = String(err.message ?? err);
    $("b-start").disabled = false;
    stopSensors();
  }
}

/** Lighter tracking for phones that cannot keep up: Firefox on Android
 * took 68-120 ms a frame while tracking (Chrome on a Nothing Phone (4a) Pro,
 * 17-45 ms with its frontend on the GPU), 7-10 fps. Fewer points (a coarser feature grid), fewer solver
 * iterations, and the estimator on every other frame (the IMU carries the
 * pose between). Natively that is 2.4x cheaper a frame on the phone
 * recordings, with fewer resets (10 against 15) but less accuracy on
 * ADVIO's long walks (3.7 m against 2.6); README "Firefox and slower
 * phones". */
const LIGHT = { cellsAcross: 10, maxIterations: 3, estimateEvery: 2 };
const lightMode = () => prefs.light === "on" || (prefs.light === "auto" && device?.browser === "firefox");

/** Whether every frame of a 60 fps camera is tracked. */
const tracking60 = () => prefs.fps60 && prefs.track60;

/** At 60 fps an exposure lasts at most 1/60 s, half the blur of a fast move
 * at 30 fps; the tracker gets every other frame unless tracking at 60 Hz
 * (see onVideoFrame). */
const videoConstraints = () => ({
  width: { ideal: 640 },
  height: { ideal: 480 },
  frameRate: prefs.fps60 ? { ideal: 60, max: 60 } : { ideal: 30, max: 30 },
});

/** The tracker needs the one lens its focal length describes, the main
 * camera. iPhones also offer virtual multi-camera devices ("Back Triple
 * Camera", "Back Dual Wide Camera") that start on, or switch to, lenses with
 * other fields of view: take "Back Camera". Android browsers name back
 * cameras "camera <id>, facing back" (Chrome) or "Facing back:<id>"
 * (Firefox), and facingMode picks whichever comes first, on a Galaxy Note20
 * Ultra the 13 mm ultra-wide "camera 2": take the lowest id, the main
 * camera on Samsung, Pixel and Nothing phones. */
async function preferMainCamera(current) {
  const track = current.getVideoTracks()[0];
  let cameras = [];
  try {
    cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  } catch (err) {
    emit("error", { where: "camera", message: `enumerateDevices: ${text(err)}` });
  }
  const backId = (label) => {
    const m = /facing back.*?(\d+)|(\d+).*facing back/i.exec(label);
    return m ? Number(m[1] ?? m[2]) : null;
  };
  const androidBacks = cameras.filter((d) => backId(d.label) !== null).sort((a, b) => backId(a.label) - backId(b.label));
  const main = cameras.find((d) => d.label === "Back Camera") ?? androidBacks[0];
  const swap = !!main && !!track && track.label !== main.label;
  emit("cameras", { labels: cameras.map((d) => d.label), current: track?.label ?? null, switchTo: swap ? main.label : null });
  if (!swap) return current;
  current.getTracks().forEach((t) => t.stop());
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: false, video: { deviceId: { exact: main.deviceId }, ...videoConstraints() } });
  } catch (err) {
    emit("error", { where: "camera", message: `main camera: ${text(err)}` });
    return navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, ...videoConstraints() } });
  }
}

function waitFor(cond, ms, what) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = () => {
      if (cond()) return resolve();
      if (performance.now() - t0 > ms) return reject(new Error(`Timed out: ${what}.`));
      setTimeout(tick, 50);
    };
    tick();
  });
}

async function startDemo() {
  $("start-err").textContent = "";
  $("b-demo").disabled = true;
  try {
    const name = new URLSearchParams(location.search).get("demo") || "advio-02";
    demo = await (await fetch(`demo/${name}.json`)).json();
    demo.next = 0;
    const video = $("video");
    video.srcObject = null;
    video.src = `demo/${name}.mp4`;
    video.loop = false;
    if (!workerReady) await waitFor(() => workerReady, 15000, "the WebAssembly module did not load");
    await video.play();
    await waitFor(() => video.videoWidth > 0, 8000, "the demo video did not load");
    video.onended = () => {
      setStatus("idle", "Demo finished", `${distance.toFixed(1)} m tracked`);
      running = false;
      $("b-stop").textContent = "Back";
    };
    $("start-sheet").hidden = true;
    $("controls").hidden = false;
    $("metrics").hidden = false;
    $("b-rec").disabled = true;
    $("map-wrap").hidden = !mapVisible;
    resizeAll();
    beginSession("demo");
    toast(demo.source, 3500);
  } catch (err) {
    console.log(err);
    emit("start-failed", { demo: true, name: err?.name, message: text(err) });
    demo = null;
    $("start-err").textContent = String(err.message ?? err);
    $("b-demo").disabled = false;
  }
}

/** Demo mode: queues the recorded IMU samples up to `t` (s). */
function feedDemoImu(t) {
  while (demo.next < demo.imu.length && demo.imu[demo.next][0] <= t) {
    const [ts, gx, gy, gz, ax, ay, az] = demo.imu[demo.next++];
    const entry = { t: ts * 1000, g: [gx, gy, gz], a: [ax, ay, az] };
    imuHistory.push(entry);
    if (imuHistory.length > 240) imuHistory.splice(0, imuHistory.length - 240);
    if (imuOnly) integrateImu(entry);
    imuBatch.push(Math.round(ts * 1e9), gx, gy, gz, ax, ay, az);
  }
  imuRateHz = demo.rig.imuRateHz;
  flushImu();
}

/** (Re)starts the VIO with the current calibration and frame geometry;
 * `reason` is for the telemetry. */
function beginSession(reason) {
  lastSync = null;
  recentSync.length = 0;
  stopImuOnly("restart");
  const video = $("video");
  const vw = video.videoWidth, vh = video.videoHeight;
  screenAngle = currentScreenAngle();
  if (demo) {
    // The recording's own rig, at the video's size.
    // The recording's own rig with this page's bias random walks.
    rig = { ...demo.rig, accelBiasWalk: IMU_DEFAULT.accelBiasWalk, gyroBiasWalk: IMU_DEFAULT.gyroBiasWalk };
    frameW = rig.width;
    frameH = rig.height;
    demo.next = 0;
    imuBatch = [];
  } else {
    // The tracked frame: `prefs.width` across the short side.
    const scale = prefs.width / Math.min(vw, vh);
    frameW = Math.round(vw * scale);
    frameH = Math.round(vh * scale);
    accelSign = decideAccelSign();
    const k = intrinsics(frameW, frameH, calib.focal35);
    const n = calib.noiseScale;
    rig = {
      width: frameW, height: frameH, ...k,
      distortion: [0, 0, 0, 0],
      imuFromCamera: [...imuFromCameraRotation(screenAngle), ...calib.cameraOffset],
      // From the learned event period: the rate counter's first window
      // spans the time before the sensors started.
      imuRateHz: Math.max(20, Math.round(motionPeriod > 0 ? 1000 / motionPeriod : imuRateHz || 60)),
      accelNoise: IMU_DEFAULT.accelNoise * n,
      gyroNoise: IMU_DEFAULT.gyroNoise * n,
      accelBiasWalk: IMU_DEFAULT.accelBiasWalk * n,
      gyroBiasWalk: IMU_DEFAULT.gyroBiasWalk * n,
      // Frames tracked per second: the estimator's defaults differ.
      frameRateHz: tracking60() ? 60 : 30,
    };
  }
  capture.width = frameW;
  capture.height = frameH;
  const light = lightMode();
  const overrides = light
    ? { "config.optical_flow_detection_grid_size": Math.round(Math.min(frameW, frameH) / LIGHT.cellsAcross), "config.vio_max_iterations": LIGHT.maxIterations }
    : undefined;
  worker.postMessage({
    type: "start", rig, gpu: prefs.gpu, overrides, estimateEvery: light ? LIGHT.estimateEvery : 1,
    watchdog: { enabled: prefs.watchdog, stillStart: prefs.stillStart },
  });
  if (!demo) {
    // The last 0.6 s of motion samples: the first frame's attitude averages
    // the half second before it (src/vio.rs, averaged_gravity).
    const since = (lastMotion?.t ?? 0) - 600;
    const recent = imuHistory.filter((s) => s.t > since);
    imuBatch = [...recent.flatMap((s) => [Math.round(s.t * 1e6), ...s.g, ...s.a]), ...imuBatch];
    flushImu();
  }
  resetView();
  running = true;
  inflight = 0;
  lastPostedTime = -Infinity;
  diag.sessionAt = performance.now();
  diag.sessionResults = 0;
  stall = null;
  setStatus("init", ...(demo ? ["Initializing", "demo: ADVIO iPhone walk"] : startMessage()));
  // One chain of frame callbacks, however often the session restarts.
  if (videoFrameCallback && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(videoFrameCallback);
  if (frameRequest) cancelAnimationFrame(frameRequest);
  scheduleFrame();
  emit("session", {
    reason, demo: !!demo, rig, light: lightMode(), accelSign, screenAngle, frame: [frameW, frameH], video: [vw, vh],
    settings: stream?.getVideoTracks()[0]?.getSettings?.() ?? null, calib, prefs,
    imuRateHz: Math.round(imuRateHz), motionPeriod, motionClock, workerReady,
  });
}

function stopSensors() {
  running = false;
  window.removeEventListener("devicemotion", onMotion);
  window.removeEventListener("deviceorientation", onOrientation);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

function stop(reason) {
  stopImuOnly("stop");
  if (running) {
    emit("stop", { reason, ...snapshot() });
    flush();
  }
  if (recording) finishRecording();
  stopSensors();
  imuBatch = []; // a later session (the demo) runs on another clock
  if (demo) {
    const video = $("video");
    video.pause();
    video.removeAttribute("src");
    video.load();
    demo = null;
    $("b-rec").disabled = false;
    $("b-demo").disabled = false;
    $("b-stop").textContent = "Stop";
  }
  worker.postMessage({ type: "stop" });
  $("controls").hidden = true;
  $("metrics").hidden = true;
  $("map-wrap").hidden = true;
  $("start-sheet").hidden = false;
  $("b-start").disabled = false;
  setStatus("idle", "Stopped");
  describeDevice($("device"));
}

function resetView() {
  trailCount = 0;
  distance = 0;
  lastTrailPos = null;
  cubePlaced = false;
  cube.visible = false;
  latest = null;
  trackAge.clear();
  landmarkGeometry.setDrawRange(0, 0);
  arLandmarkGeometry.setDrawRange(0, 0);
  trailGeometry.setDrawRange(0, 0);
  imuTrailCount = 0;
  imuTrailGeometry.setDrawRange(0, 0);
}

// ---------------------------------------------------------------- frames

function scheduleFrame() {
  const video = $("video");
  if (!running) return;
  if (video.requestVideoFrameCallback) {
    videoFrameCallback = video.requestVideoFrameCallback(onVideoFrame);
  } else {
    frameRequest = requestAnimationFrame((now) => onVideoFrame(now, null));
  }
}

// Frame times on the performance.now() clock: the capture time where the
// browser reports it (Chrome), else when the frame was presented, else the
// callback's time. A source that is missing, more than a second from now or
// not increasing is given up for the next one, for good.
const FRAME_CLOCKS = ["capture", "presentation", "callback"];
let frameClock = 0;
let lastClockT = -Infinity;
function frameTime(now, meta) {
  const perf = performance.now();
  for (;;) {
    const name = FRAME_CLOCKS[frameClock];
    const t = name === "capture" ? meta?.captureTime : name === "presentation" ? meta?.presentationTime : now;
    if (name === "callback" || (Number.isFinite(t) && Math.abs(t - perf) < 1000 && t > lastClockT)) {
      lastClockT = t;
      return t;
    }
    frameClock++;
    emit("frame-clock", { from: name, to: FRAME_CLOCKS[frameClock], value: t ?? null, last: Number.isFinite(lastClockT) ? lastClockT : null, perf, now });
  }
}

/** rVFC metadata, numbers rounded, for the telemetry. */
function metaFields(meta) {
  if (!meta) return null;
  const out = {};
  for (const k in meta) out[k] = typeof meta[k] === "number" ? Math.round(meta[k] * 1000) / 1000 : meta[k];
  return out;
}

// Each callback should bring a new frame: repeats (same frame count and
// media time) are skipped, unless the browser never advances either.
let lastFrameKey = null, repeatedFrames = 0, skipRepeats = true;
function repeatedFrame(meta) {
  if (!meta || !skipRepeats) return false;
  const key = `${meta.presentedFrames}/${meta.mediaTime}`;
  if (key !== lastFrameKey) {
    lastFrameKey = key;
    repeatedFrames = 0;
    return false;
  }
  diag.duplicates++;
  if (++repeatedFrames > 30) {
    skipRepeats = false;
    emit("repeats-off", { key, meta: metaFields(meta) });
  }
  return true;
}

function onVideoFrame(now, meta) {
  if (!running) return;
  scheduleFrame();
  diag.callbacks++;
  diag.lastCallbackAt = performance.now();
  if (repeatedFrame(meta)) return;
  let t;
  if (demo) {
    // The recorded frame time of this video frame; its IMU samples (and a
    // little beyond, which closes the interval) go first.
    const n = Math.min(demo.frameTimes.length - 1, Math.round((meta?.mediaTime ?? $("video").currentTime) * demo.fps));
    t = demo.frameTimes[n] * 1000;
    feedDemoImu(demo.frameTimes[n] + 0.05);
  } else {
    t = frameTime(now, meta) + calib.timeOffsetMs;
  }
  if (diag.framesLogged < 8 || diag.callbacks % 300 === 0) {
    diag.framesLogged++;
    const r = (v) => Math.round(v * 1000) / 1000;
    emit("frame-meta", {
      callbacks: diag.callbacks, t: r(t), now: r(now), perf: r(performance.now()), clock: demo ? "demo" : FRAME_CLOCKS[frameClock],
      imuT: lastMotion && !demo ? r(lastMotion.t) : null, inflight, meta: metaFields(meta),
    });
  }
  lastFrameTime = t;
  if (!demo && Number.isFinite(meta?.captureTime)) {
    const latency = performance.now() - meta.captureTime;
    if (latency > 0 && latency < 200) videoLatencyMs += 0.05 * (latency - videoLatencyMs);
  }
  // At 60 fps only every other frame is tracked (unless tracking at 60
  // Hz); a recording keeps them all.
  const track = !(prefs.fps60 && !tracking60() && !demo && t - lastPostedTime < 25);
  if (!track) diag.skippedRate++;
  else if (inflight >= 2) diag.skippedBusy++; // the worker is behind
  if (!recording && (!track || inflight >= 2)) return;
  const grabStart = performance.now();
  captureCtx.drawImage($("video"), 0, 0, frameW, frameH);
  const rgba = captureCtx.getImageData(0, 0, frameW, frameH).data;
  const gray = new Uint8Array(frameW * frameH);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = (77 * rgba[j] + 150 * rgba[j + 1] + 29 * rgba[j + 2]) >> 8;
  }
  const grabMs = performance.now() - grabStart;
  grabMsAvg = grabMsAvg ? 0.9 * grabMsAvg + 0.1 * grabMs : grabMs;
  if (diag.callbacks % 15 === 0) frameLevel = greyLevel(gray);
  if (recording) {
    recording.frames.push(gray.slice());
    recording.times.push(Math.round(t * 1e6));
    const secs = (performance.now() - recording.started) / 1000;
    $("m-rec-t").textContent = secs.toFixed(0);
    if (recording.frames.length * gray.length > REC_MAX_BYTES) finishRecording();
  }
  if (!track || inflight >= 2) return;
  flushImu();
  inflight++;
  diag.posted++;
  lastPostedTime = t;
  worker.postMessage({ type: "frame", t: Math.round(t * 1e6), gray: gray.buffer, width: frameW, height: frameH }, [gray.buffer]);
}

// ---------------------------------------------------------------- results

/** Whether a result's pose is a tracked one (not waiting or initializing;
 * coasting on the IMU counts). */
const posed = (status) => status !== "init" && status !== "wait";

/** The status line while a session starts. */
const startMessage = () =>
  prefs.stillStart ? ["Hold still", "then move the phone slowly side to side"] : ["Initializing", "move the phone side to side"];

function onResult(m) {
  latest = m;
  results++;
  msAvg = msAvg ? 0.9 * msAvg + 0.1 * m.ms : m.ms;
  if (m.gpuWaitMs != null) {
    gpuWaitAvg = gpuWaitAvg ? 0.9 * gpuWaitAvg + 0.1 * m.gpuWaitMs : m.gpuWaitMs;
    gpuSubmitAvg = gpuSubmitAvg ? 0.9 * gpuSubmitAvg + 0.1 * m.gpuSubmitMs : m.gpuSubmitMs;
    finishAvg = finishAvg ? 0.9 * finishAvg + 0.1 * m.finishMs : m.finishMs;
  }
  const now = performance.now();
  diag.results++;
  diag.sessionResults++;
  diag.lastResultAt = now;
  if (stall) {
    emit("unstall", { ms: Math.round(now - stall.at), reason: stall.reason });
    stall = null;
  }
  if (diag.sessionResults <= 5 || m.status !== lastStatus) {
    emit("result", {
      n: diag.sessionResults, frameId: m.frameId, t: m.t, status: m.status, connected: m.connected, tracks: m.trackIds.length,
      imuSamples: m.imuSamples, keyframe: m.keyframe, ms: Math.round(m.ms), resets: m.resets,
    });
    lastStatus = m.status;
  }
  if (now - resultsWindow > 1000) {
    fps = (1000 * results) / (now - resultsWindow);
    results = 0;
    resultsWindow = now;
  }
  const p = [m.camera[4], m.camera[5], m.camera[6]];
  if (imuOnly) {
    compareImuOnly(m);
  } else if (m.status.startsWith("reset")) {
    setStatus("reset", "Lost tracking", `restarting (${m.status.slice(6)})`);
    toast(prefs.stillStart ? "Tracking diverged: hold the phone still to restart." : "Tracking diverged: re-initializing. Move side to side.");
  } else if (m.status === "wait") {
    setStatus("init", "Hold still", "tracking starts once the phone is still");
  } else if (m.status === "init") {
    setStatus("init", "Initializing", "move the phone slowly side to side");
  } else if (m.status === "coast") {
    setStatus("coast", "Coasting on IMU", "the camera lost its points: slow down");
  } else {
    setStatus("track", "Tracking", `${m.connected} landmarks`);
    if (!cubePlaced) placeCube(m);
  }
  if (posed(m.status)) {
    if (lastTrailPos) distance += Math.hypot(p[0] - lastTrailPos[0], p[1] - lastTrailPos[1], p[2] - lastTrailPos[2]);
    lastTrailPos = p;
    if (trailCount < trail.length / 3) {
      trail.set(p, 3 * trailCount++);
      trailGeometry.attributes.position.needsUpdate = true;
      trailGeometry.setDrawRange(0, trailCount);
    }
  }
  if (m.landmarks && !imuOnly) {
    const n = Math.min(m.landmarks.length / 3, MAX_LANDMARKS);
    landmarkPositions.set(m.landmarks.subarray(0, 3 * n));
    // Two geometries (one per renderer) over the same array.
    landmarkGeometry.attributes.position.needsUpdate = true;
    arLandmarkGeometry.attributes.position.needsUpdate = true;
    landmarkGeometry.setDrawRange(0, n);
    arLandmarkGeometry.setDrawRange(0, n);
  }
  for (let i = 0; i < m.trackIds.length; i++) {
    const id = m.trackIds[i];
    if (!trackAge.has(id)) trackAge.set(id, m.frameId);
  }
  if (trackAge.size > 4000) trackAge.clear();
  if (m.sync) onSync(m.sync);
  const v = Math.hypot(m.velocity[0], m.velocity[1], m.velocity[2]);
  $("m-fps").textContent = fps.toFixed(0);
  $("m-ms").textContent = msAvg.toFixed(0);
  $("m-pts").textContent = `${m.connected}/${m.trackIds.length}`;
  $("m-imu").textContent = imuRateHz.toFixed(0);
  $("m-dist").textContent = `${distance.toFixed(1)} m`;
  $("m-speed").textContent = `${v.toFixed(2)} m/s`;
  mapDirty = true;
}

/** A time-offset measurement from the session: when it is solid and the
 * frames are off by more than a few milliseconds, shift the frame
 * timestamps, save the offset for this phone and restart tracking. */
function onSync(e) {
  lastSync = e;
  emit("sync", { ...e, imuFromCamera: Array.from(e.imuFromCamera, (v) => Math.round(v * 1e4) / 1e4) });
  if (!$("settings-sheet").hidden) fillSyncInfo();
  if (demo || checkGyroAxes(e)) return;
  // Moving through space (not just turning) makes single estimates swing by
  // 10-15 ms (correlation 0.91-0.92 on an iPhone), and every correction
  // restarts tracking: only correct on solid estimates that agree, and only
  // by 15 ms or more. Even solid ones read +-8 ms off depending on the
  // motion: an iPhone's saved offset went -30 -> -22 -> -30 ms over two
  // sessions, a restart each, where a bundle adjustment put it at -26.
  recentSync.push(e.offsetMs);
  if (recentSync.length > 3) recentSync.shift();
  const steady = recentSync.length === 3 && Math.max(...recentSync) - Math.min(...recentSync) < 4;
  const solid = e.correlation > 0.93 && e.intervals >= 250 && e.rateRadS > 0.15 && steady;
  if (!solid || Math.abs(e.offsetMs) < 15 || syncApplied >= 2) return;
  calib.timeOffsetMs = Math.round(calib.timeOffsetMs + e.offsetMs);
  saveCalibration(device, calib);
  syncApplied++;
  toast(`Synchronized camera and motion sensors: ${e.offsetMs > 0 ? "+" : ""}${e.offsetMs.toFixed(0)} ms (saved for this phone)`, 3500);
  emit("sync-applied", { ...e, timeOffsetMs: calib.timeOffsetMs });
  beginSession("sync");
}

function fillSyncInfo() {
  const el = $("sync-info");
  if (!lastSync) {
    el.textContent = "Camera-IMU sync: measuring (turn the phone around a bit while tracking)…";
    return;
  }
  // The focal scale and rotation it also reports assume pure rotation: a
  // hand-held phone's translation biases them (on iPhone recordings the
  // focal read 7-13 % and the rotation 7-18° off, against a bundle
  // adjustment that found the profile right within 2 % and 1°), so they are
  // only good for catching gross errors (a wrong lens, swapped axes).
  const e = lastSync;
  el.textContent = `Camera-IMU sync${demo ? " (demo recording)" : ""}: residual ${e.offsetMs.toFixed(1)} ms ` +
    `(correlation ${e.correlation.toFixed(2)}). Rough checks: focal x${e.focalScale.toFixed(2)}, ` +
    `camera-IMU rotation within ${e.rotationErrorDeg.toFixed(0)}° of the assumed one.`;
}

function placeCube(m, ahead = 1) {
  // `ahead` m in front of the camera along its horizontal heading, a little below.
  const q = new THREE.Quaternion(m.camera[1], m.camera[2], m.camera[3], m.camera[0]);
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
  fwd.z = 0;
  if (fwd.lengthSq() < 1e-4) fwd.set(1, 0, 0);
  fwd.normalize();
  cube.position.set(m.camera[4] + ahead * fwd.x, m.camera[5] + ahead * fwd.y, m.camera[6] - 0.25);
  cube.rotation.set(0, 0, Math.atan2(fwd.y, fwd.x));
  cube.visible = true;
  cubePlaced = true;
}

// ---------------------------------------------------------------- AR view

const arCanvas = $("ar");
const arRenderer = new THREE.WebGLRenderer({ canvas: arCanvas, alpha: true, antialias: true });
arRenderer.setClearColor(0x000000, 0);
const arScene = new THREE.Scene();
const arCamera = new THREE.PerspectiveCamera();
arCamera.matrixAutoUpdate = false;

const cube = new THREE.Group();
{
  const s = 0.2;
  const box = new THREE.BoxGeometry(s, s, s);
  box.translate(0, 0, s / 2);
  cube.add(new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color: 0xe0835c, transparent: true, opacity: 0.35, depthWrite: false })));
  cube.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0xffd2bd })));
  const axes = new THREE.AxesHelper(0.3);
  cube.add(axes);
  cube.visible = false;
}
const MAX_LANDMARKS = 4000;
const landmarkPositions = new Float32Array(3 * MAX_LANDMARKS);
const arLandmarkGeometry = new THREE.BufferGeometry();
arLandmarkGeometry.setAttribute("position", new THREE.BufferAttribute(landmarkPositions, 3));
arLandmarkGeometry.setDrawRange(0, 0);
const arLandmarks = new THREE.Points(arLandmarkGeometry, new THREE.PointsMaterial({ color: 0x6cc48a, size: 6, sizeAttenuation: false, depthTest: false }));
arScene.add(cube, arLandmarks);

/** The camera's pose at `t` (ms on the IMU clock; see overlayTime): the
 * last VIO pose carried forward by the gyroscope
 * (bias-corrected, the newest rate held past the newest sample) and by its
 * velocity, with the camera's offset from the IMU turned by the new
 * attitude. The overlay then matches the image it is drawn over, not the
 * older frame the tracker finished. */
function predictedCameraPose(t = lastFrameTime) {
  const m = latest;
  if (RAW_POSE) return { q: [m.camera[0], m.camera[1], m.camera[2], m.camera[3]], p: [m.camera[4], m.camera[5], m.camera[6]] };
  const tf = m.t / 1e6;
  // Not before the estimate; past the newest motion sample, the newest
  // rate is held (an extrapolation of a few milliseconds).
  const target = Math.max(tf, Math.min(t, tf + 250));
  let qi = [m.imu[0], m.imu[1], m.imu[2], m.imu[3]];
  const bg = m.gyroBias;
  const turn = (g, h) => {
    qi = quatMul(qi, quatFromRotvec([(g[0] - bg[0]) * h, (g[1] - bg[1]) * h, (g[2] - bg[2]) * h]));
  };
  let covered = tf;
  for (let k = 1; k < imuHistory.length; k++) {
    const a = imuHistory[k - 1], b = imuHistory[k];
    if (b.t <= tf || a.t >= target) continue;
    const t0 = Math.max(a.t, tf), t1 = Math.min(b.t, target);
    turn(b.g, (t1 - t0) / 1000);
    covered = t1;
  }
  const newest = imuHistory[imuHistory.length - 1];
  if (newest && covered < target) turn(newest.g, (target - covered) / 1000);
  const dt = (target - tf) / 1000;
  // T_w_c = T_w_i * T_i_c.
  const qc = quatMul(qi, rig.imuFromCamera.slice(0, 4));
  const offset = quatRotate(qi, rig.imuFromCamera.slice(4, 7));
  const pos = [0, 1, 2].map((i) => m.imu[4 + i] + m.velocity[i] * dt + offset[i]);
  return { q: qc, p: pos };
}

/** Projection of the tracked frame's pinhole camera onto the screen, where
 * the video is shown with object-fit: cover. */
function screenIntrinsics(sw, sh) {
  const video = $("video");
  const vw = video.videoWidth, vh = video.videoHeight;
  const cover = Math.max(sw / vw, sh / vh);
  const ox = (sw - vw * cover) / 2, oy = (sh - vh * cover) / 2;
  const k = cover * (vw / frameW); // tracked-frame pixel -> screen px
  return { fx: rig.fx * k, fy: rig.fy * k, cx: ox + (rig.cx + 0.5) * k, cy: oy + (rig.cy + 0.5) * k, k, ox, oy };
}

function renderAR(t = lastFrameTime) {
  const sw = arCanvas.clientWidth, sh = arCanvas.clientHeight;
  if (!rig || !(imuOnly || (latest && posed(latest.status)))) {
    arRenderer.clear();
    return;
  }
  const { fx, fy, cx, cy } = screenIntrinsics(sw, sh);
  const n = 0.05, f = 100;
  const P = new THREE.Matrix4().set(
    (2 * fx) / sw, 0, 1 - (2 * cx) / sw, 0,
    0, (2 * fy) / sh, (2 * cy) / sh - 1, 0,
    0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n),
    0, 0, -1, 0,
  );
  arCamera.projectionMatrix.copy(P);
  arCamera.projectionMatrixInverse.copy(P).invert();
  const { q, p } = imuOnly ? imuOnlyCameraPose(lastFrameTime) : predictedCameraPose(t);
  // OpenCV camera (x right, y down, z ahead) -> three.js camera (y up,
  // looking down -z): a half turn about x.
  const qc = new THREE.Quaternion(q[1], q[2], q[3], q[0]).multiply(new THREE.Quaternion(1, 0, 0, 0));
  arCamera.matrix.compose(new THREE.Vector3(p[0], p[1], p[2]), qc, new THREE.Vector3(1, 1, 1));
  arCamera.matrixWorld.copy(arCamera.matrix);
  arCamera.matrixWorldInverse.copy(arCamera.matrix).invert();
  arLandmarks.visible = prefs.landmarks;
  arRenderer.render(arScene, arCamera);
}

// Tracked points over the camera image.
const trackCanvas = $("tracks");
const trackCtx = trackCanvas.getContext("2d");
function renderTracks(t = lastFrameTime) {
  const sw = trackCanvas.clientWidth, sh = trackCanvas.clientHeight;
  trackCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  trackCtx.clearRect(0, 0, sw, sh);
  // Tracked points are camera information, from the frame the tracker
  // finished: shown only on request (⚙), and then turned onto the frame on
  // screen by the rotation in between (as points at infinity).
  if (!prefs.tracks || !latest || !rig || imuOnly) return;
  const { k, ox, oy } = screenIntrinsics(sw, sh);
  const shown = predictedCameraPose(t);
  const tracked = [latest.camera[0], latest.camera[1], latest.camera[2], latest.camera[3]];
  const toShown = quatMul(quatConj(shown.q), tracked);
  const xy = latest.trackXY, ids = latest.trackIds;
  for (let i = 0; i < ids.length; i++) {
    const age = latest.frameId - (trackAge.get(ids[i]) ?? latest.frameId);
    const b = quatRotate(toShown, [(xy[2 * i] - rig.cx) / rig.fx, (xy[2 * i + 1] - rig.cy) / rig.fy, 1]);
    if (b[2] <= 0) continue;
    const u = rig.fx * (b[0] / b[2]) + rig.cx, v = rig.fy * (b[1] / b[2]) + rig.cy;
    const x = ox + (u + 0.5) * k, y = oy + (v + 0.5) * k;
    trackCtx.fillStyle = age > 20 ? "rgba(108,196,138,0.9)" : age > 5 ? "rgba(230,180,80,0.9)" : "rgba(229,101,91,0.9)";
    trackCtx.beginPath();
    trackCtx.arc(x, y, 2.5, 0, 2 * Math.PI);
    trackCtx.fill();
  }
}

// ---------------------------------------------------------------- IMU only

// The camera ignored: the pose comes from integrating the motion samples
// from the last tracked state (gyroscope for the attitude; accelerometer
// less gravity for velocity and position; both bias-corrected) and the map
// is frozen. If the frozen points stay on their scene features while the
// phone turns, the gyroscope axes, camera-IMU rotation, focal length and
// time offset agree; position drifts within seconds, as double-integrated
// accelerometers do. The VIO keeps tracking behind, for comparison.
const GRAVITY = 9.81; // m/s² along -z, as the estimator's world
let imuOnly = null; // { t, q, p, v, bg, ba, g, a, poses: [{ t, q, p }], startT, resets, drift }
const imuTrail = new Float32Array(3 * 20000);
let imuTrailCount = 0;

/** `v` rotated by the unit quaternion `q` = [w, x, y, z]. */
function quatRotate(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx];
}

function startImuOnly() {
  const m = latest;
  if (!running || !m) {
    toast("Start tracking first");
    return;
  }
  const t = m.t / 1e6;
  const q = Array.from(m.imu.slice(0, 4)), p = Array.from(m.imu.slice(4, 7));
  imuOnly = {
    t, q, p, v: Array.from(m.velocity), bg: Array.from(m.gyroBias), ba: Array.from(m.accelBias),
    poses: [{ t, q: q.slice(), p: p.slice() }], startT: t, resets: m.resets, drift: null, reportedAt: 0,
  };
  imuTrailCount = 0;
  // From the tracked frame's time to the newest sample, starting from the
  // sample before it.
  const before = imuHistory.filter((sample) => sample.t <= t).pop();
  if (before) unbias(imuOnly, before);
  for (const sample of imuHistory) if (sample.t > t) integrateImu(sample);
  if (!cubePlaced) {
    const c = imuOnlyCameraPose(imuOnly.t);
    placeCube({ camera: [...c.q, ...c.p] });
  }
  $("b-imu").setAttribute("aria-pressed", "true");
  setStatus("imu", "IMU only", "camera ignored");
  emit("imu-only", {
    on: true, status: m.status, frameId: m.frameId, connected: m.connected,
    v: imuOnly.v, gyroBias: imuOnly.bg, accelBias: imuOnly.ba, accelSign, rig,
  });
}

function stopImuOnly(reason) {
  if (!imuOnly) return;
  const st = imuOnly;
  imuOnly = null;
  $("b-imu").setAttribute("aria-pressed", "false");
  emit("imu-only", { on: false, reason, s: Math.round(st.t - st.startT) / 1000, drift: st.drift });
  mapDirty = true;
}

/** A sample's bias-corrected rate and specific force, kept as the start of
 * the next interval. */
function unbias(st, sample) {
  st.g = [0, 1, 2].map((k) => sample.g[k] - st.bg[k]);
  st.a = [0, 1, 2].map((k) => sample.a[k] - st.ba[k]);
}

/** One motion sample into the IMU-only state: trapezoids between it and
 * the previous one (mean rate for the attitude, mean world acceleration for
 * velocity and position). */
function integrateImu(sample) {
  const st = imuOnly;
  const h = Math.min(0.1, (sample.t - st.t) / 1000);
  if (!(h > 0)) return;
  const [g0, a0] = st.g ? [st.g, st.a] : [null, null];
  unbias(st, sample);
  const w0 = g0 ?? st.g, f0 = quatRotate(st.q, a0 ?? st.a);
  const q = quatMul(st.q, quatFromRotvec([0, 1, 2].map((k) => 0.5 * (w0[k] + st.g[k]) * h)));
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  const q1 = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  const f1 = quatRotate(q1, st.a);
  const acc = [0.5 * (f0[0] + f1[0]), 0.5 * (f0[1] + f1[1]), 0.5 * (f0[2] + f1[2]) - GRAVITY];
  for (let k = 0; k < 3; k++) {
    st.p[k] += st.v[k] * h + 0.5 * acc[k] * h * h;
    st.v[k] += acc[k] * h;
  }
  st.q = q1;
  st.t = sample.t;
  st.poses.push({ t: st.t, q: st.q.slice(), p: st.p.slice() });
  if (st.poses.length > 900) st.poses.splice(0, st.poses.length - 900);
  if (imuTrailCount < imuTrail.length / 3) {
    imuTrail.set(st.p, 3 * imuTrailCount++);
    imuTrailGeometry.attributes.position.needsUpdate = true;
    imuTrailGeometry.setDrawRange(0, imuTrailCount);
  }
  mapDirty = true;
}

/** The IMU-only IMU pose at `t` (ms): the last one integrated by then. */
function imuOnlyPoseAt(t) {
  const poses = imuOnly.poses;
  for (let i = poses.length - 1; i > 0; i--) if (poses[i].t <= t) return poses[i];
  return poses[0];
}

/** T_w_c from the IMU-only state at `t` (ms). */
function imuOnlyCameraPose(t) {
  const s = imuOnlyPoseAt(t);
  const ic = rig.imuFromCamera;
  const lever = quatRotate(s.q, [ic[4], ic[5], ic[6]]);
  return { q: quatMul(s.q, ic.slice(0, 4)), p: [s.p[0] + lever[0], s.p[1] + lever[1], s.p[2] + lever[2]] };
}

/** The status while IMU only: how far the integrated pose is from the VIO's
 * (still tracking behind) at the VIO's newest frame. */
function compareImuOnly(m) {
  const st = imuOnly;
  let sub = `camera ignored · ${((st.t - st.startT) / 1000).toFixed(1)} s`;
  if (m.resets !== st.resets) {
    sub += " · tracking restarted, no comparison";
  } else {
    const s = imuOnlyPoseAt(m.t / 1e6);
    const dot = Math.abs(s.q[0] * m.imu[0] + s.q[1] * m.imu[1] + s.q[2] * m.imu[2] + s.q[3] * m.imu[3]);
    const deg = (2 * Math.acos(Math.min(1, dot))) / DEG;
    const metres = Math.hypot(s.p[0] - m.imu[4], s.p[1] - m.imu[5], s.p[2] - m.imu[6]);
    st.drift = { s: Math.round(m.t / 1e6 - st.startT) / 1000, deg: Math.round(deg * 100) / 100, m: Math.round(metres * 1000) / 1000 };
    sub += ` · vs tracking ${deg.toFixed(1)}°, ${metres.toFixed(2)} m`;
    const now = performance.now();
    if (now - st.reportedAt > 1000) {
      st.reportedAt = now;
      emit("imu-only-drift", { ...st.drift, status: m.status, connected: m.connected });
    }
  }
  setStatus("imu", "IMU only", sub);
}

// ---------------------------------------------------------------- map view

const mapCanvas = $("map");
const mapRenderer = new THREE.WebGLRenderer({ canvas: mapCanvas, antialias: true });
mapRenderer.setClearColor(0x0e1012, 1);
mapRenderer.setPixelRatio(Math.min(2, devicePixelRatio));
const mapScene = new THREE.Scene();
const mapCamera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
mapCamera.up.set(0, 0, 1);
mapCamera.position.set(-3, -3, 4);
const controls = new OrbitControls(mapCamera, mapCanvas);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
let follow = true;
let mapDirty = true;
let mapVisible = localStorage.getItem("visloc-vio-map") !== "0";
controls.addEventListener("start", () => setFollow(false));
controls.addEventListener("change", () => (mapDirty = true));

const grid = new THREE.GridHelper(40, 40, 0x3a4048, 0x23272c);
grid.rotation.x = Math.PI / 2;
grid.position.z = -1.4;
const trailGeometry = new THREE.BufferGeometry();
trailGeometry.setAttribute("position", new THREE.BufferAttribute(trail, 3));
trailGeometry.setDrawRange(0, 0);
const trailLine = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({ color: 0xe0835c }));
trailLine.frustumCulled = false;
const landmarkGeometry = new THREE.BufferGeometry();
landmarkGeometry.setAttribute("position", new THREE.BufferAttribute(landmarkPositions, 3));
landmarkGeometry.setDrawRange(0, 0);
const mapLandmarks = new THREE.Points(landmarkGeometry, new THREE.PointsMaterial({ color: 0xd8dde2, size: 3, sizeAttenuation: false }));
mapLandmarks.frustumCulled = false;
arLandmarks.frustumCulled = false;
// The phone: a small camera frustum in the camera frame (z ahead, y down).
const phone = new THREE.LineSegments(
  (() => {
    const g = new THREE.BufferGeometry();
    const d = 0.18, w = 0.1, h = 0.13;
    const c = [[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]];
    const v = [];
    for (const p of c) v.push(0, 0, 0, ...p);
    for (let i = 0; i < 4; i++) v.push(...c[i], ...c[(i + 1) % 4]);
    v.push(-w * 0.6, -h, d, 0, -h * 1.6, d, 0, -h * 1.6, d, w * 0.6, -h, d); // "up" tick
    g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    return g;
  })(),
  new THREE.LineBasicMaterial({ color: 0xffffff }),
);
const mapCube = cube.clone();
const imuTrailGeometry = new THREE.BufferGeometry();
imuTrailGeometry.setAttribute("position", new THREE.BufferAttribute(imuTrail, 3));
imuTrailGeometry.setDrawRange(0, 0);
const imuTrailLine = new THREE.Line(imuTrailGeometry, new THREE.LineBasicMaterial({ color: 0x6aa8e8 }));
imuTrailLine.frustumCulled = false;
mapScene.add(grid, trailLine, imuTrailLine, mapLandmarks, phone, mapCube, new THREE.AxesHelper(0.5));

function setFollow(on) {
  follow = on;
  $("follow").setAttribute("aria-pressed", String(on));
  $("follow").textContent = on ? "following" : "follow";
}
$("follow").onclick = () => setFollow(true);

function renderMap() {
  if (!mapVisible) return;
  const pose = imuOnly ? imuOnlyCameraPose(lastFrameTime) : latest && { q: latest.camera.slice(0, 4), p: latest.camera.slice(4, 7) };
  if (pose) {
    phone.position.set(pose.p[0], pose.p[1], pose.p[2]);
    phone.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
    grid.position.x = Math.round(pose.p[0]);
    grid.position.y = Math.round(pose.p[1]);
    if (follow) {
      const target = phone.position;
      const offset = mapCamera.position.clone().sub(controls.target);
      controls.target.lerp(target, 0.15);
      mapCamera.position.copy(controls.target).add(offset);
      mapDirty = true;
    }
  }
  mapCube.visible = cube.visible;
  mapCube.position.copy(cube.position);
  mapCube.rotation.copy(cube.rotation);
  controls.update();
  if (mapDirty) {
    mapRenderer.render(mapScene, mapCamera);
    mapDirty = false;
  }
}

// ---------------------------------------------------------------- loop & layout

function resizeAll() {
  const w = innerWidth, h = innerHeight;
  arRenderer.setPixelRatio(Math.min(2, devicePixelRatio));
  arRenderer.setSize(w, h, false);
  trackCanvas.width = Math.round(w * devicePixelRatio);
  trackCanvas.height = Math.round(h * devicePixelRatio);
  const mw = mapCanvas.clientWidth, mh = mapCanvas.clientHeight;
  if (mw && mh) {
    mapRenderer.setSize(mw, mh, false);
    mapCamera.aspect = mw / mh;
    mapCamera.updateProjectionMatrix();
  }
  mapDirty = true;
}
addEventListener("resize", () => {
  resizeAll();
  if (running && currentScreenAngle() !== screenAngle) {
    toast("Orientation changed: restarting tracking");
    beginSession("orientation");
  }
});

function loop(rafNow) {
  requestAnimationFrame(loop);
  const interval = rafNow - lastRafAt;
  lastRafAt = rafNow;
  if (interval > 4 && interval < 40) refreshMs += 0.05 * (interval - refreshMs);
  if (!running) return;
  const t = overlayTime(rafNow);
  renderTracks(t);
  renderAR(t);
  renderMap();
}

/** The capture time (ms, IMU clock) of the image on screen when an overlay
 * drawn now appears: one refresh from now, less the camera's latency to the
 * page. The video shows frames as they come, ahead of anything drawn, so
 * the newest frame the page has seen is already a frame or so behind it. In
 * the demo (a video file) frames have no capture time: the newest frame. */
function overlayTime(rafNow) {
  if (demo || !Number.isFinite(rafNow)) return lastFrameTime;
  return rafNow + refreshMs - videoLatencyMs + calib.timeOffsetMs + prefs.overlayLeadMs;
}
requestAnimationFrame(loop);

// A tab the phone resumes keeps the code it loaded: after a deploy it would
// run the old build (an iPhone hit an error already fixed that way). When
// the page comes back with the files changed, it reloads.
const BUILD_FILES = ["./", "./app.js", "./worker.js", "./profiles.js", "./pkg/visloc_wasm_bg.wasm", "./pkg-mt/visloc_wasm_bg.wasm"];
const buildStamp = () =>
  Promise.all(BUILD_FILES.map((f) => fetch(f, { method: "HEAD", cache: "no-store" }).then((r) => r.headers.get("last-modified") ?? "")))
    .then((stamps) => stamps.join("|"))
    .catch(() => null);
const loadedBuild = buildStamp();

document.addEventListener("visibilitychange", async () => {
  if (document.hidden && running) stop("hidden");
  if (document.hidden) return;
  const [loaded, now] = [await loadedBuild, await buildStamp()];
  if (loaded && now && loaded !== now && !running) {
    emit("reload", { reason: "rebuilt" });
    flush(true);
    location.reload();
  }
});

// ---------------------------------------------------------------- diagnostics

/** Where frames, motion samples and results stand, for the telemetry and
 * the ?debug overlay. Counters are cumulative; `since` is ms ago. */
function snapshot(now = performance.now()) {
  const ms = (v) => (Number.isFinite(v) ? Math.round(v) : null);
  const r = (v, k = 1000) => Math.round(v * k) / k;
  const w = workerStats, s = w?.session, m = latest;
  return {
    frames: {
      callbacks: diag.callbacks, duplicates: diag.duplicates, skippedBusy: diag.skippedBusy, skippedRate: diag.skippedRate, posted: diag.posted,
      results: diag.results, dropped: diag.dropped, errors: diag.errors, inflight,
      clock: demo ? "demo" : FRAME_CLOCKS[frameClock], size: [frameW, frameH],
    },
    motion: {
      events: diag.motion, incomplete: diag.motionIncomplete, snaps: diag.motionSnaps, sent: diag.imuSent,
      rateHz: ms(imuRateHz), periodMs: r(motionPeriod, 100), clock: motionClock,
    },
    since: {
      frame: ms(now - diag.lastCallbackAt), motion: ms(now - diag.lastMotionAt), result: ms(now - diag.lastResultAt),
      workerBeat: ms(now - workerBeatAt), session: ms(now - diag.sessionAt),
    },
    // Newest frame time minus newest motion time: > 0 when frames run ahead.
    frameAheadOfImuMs: lastMotion && !demo ? ms(lastFrameTime - lastMotion.t) : null,
    worker: w && {
      frames: w.frames, replaced: w.replaced, processed: w.processed, notReady: w.notReady, errors: w.errors,
      imuSamples: w.imuSamples, imuAccepted: w.imuAccepted, lastMs: ms(w.lastMs), maxMs: ms(w.maxMs),
      lastImuSamples: w.lastImuSamples, busy: w.busy, hasSession: w.hasSession,
      pendingAheadOfImuMs: w.pendingT != null && s?.imuLast != null ? ms((w.pendingT - s.imuLast) / 1e6) : null,
      pendingAfterLastFrameMs: w.pendingT != null && s?.lastFrame != null ? ms((w.pendingT - s.lastFrame) / 1e6) : null,
      imuQueued: s?.imuQueued ?? null, sessionFrames: s?.frames ?? null, sessionError: s?.error,
    },
    imuOnly: imuOnly && { s: Math.round(imuOnly.t - imuOnly.startT) / 1000, drift: imuOnly.drift },
    vio: m && {
      frameId: m.frameId, status: m.status, connected: m.connected, tracks: m.trackIds.length, imuSamples: m.imuSamples,
      overlay: {
        videoLatencyMs: r(videoLatencyMs, 10),
        refreshMs: r(refreshMs, 10),
        leadMs: prefs.overlayLeadMs,
        aheadOfImuMs: lastMotion ? r(overlayTime(lastRafAt) - lastMotion.t, 10) : null,
        aheadOfEstimateMs: r(overlayTime(lastRafAt) - m.t / 1e6, 10),
      },
      frameLevel,
      msAvg: ms(msAvg), grabMsAvg: r(grabMsAvg, 10), gpuWaitAvg: frontendGpu ? r(gpuWaitAvg, 10) : null, gpuSubmitAvg: frontendGpu ? r(gpuSubmitAvg, 10) : null, finishAvg: frontendGpu ? r(finishAvg, 10) : null, fps: r(fps, 10), resets: m.resets,
      p: Array.from(m.camera.slice(4, 7), (v) => r(v)), v: Array.from(m.velocity, (v) => r(v)),
      gyroBias: Array.from(m.gyroBias, (v) => r(v, 1e4)), accelBias: Array.from(m.accelBias, (v) => r(v)),
    },
  };
}

/** Why no results are arriving, as far as the page can tell. */
function stallReason(now) {
  const w = workerStats, s = w?.session;
  if (now - workerBeatAt > 2500) return `the VIO worker stopped responding ${((now - workerBeatAt) / 1000).toFixed(0)} s ago`;
  if (now - diag.lastCallbackAt > 2000) return "no camera frames are arriving";
  if (!demo && now - diag.lastMotionAt > 2000) return "no motion events are arriving";
  if (w?.pendingT != null && s) {
    if (s.lastFrame != null && w.pendingT <= s.lastFrame) return `frame times are not increasing (${((s.lastFrame - w.pendingT) / 1e6).toFixed(0)} ms behind the last frame)`;
    if (s.imuLast == null) return "the VIO has no motion samples";
    if (w.pendingT > s.imuLast) return `motion samples lag the frames by ${((w.pendingT - s.imuLast) / 1e6).toFixed(0)} ms`;
  }
  if (diag.errors) return "the VIO reported errors";
  return "no results (cause unknown)";
}

/** Every second: stall detection, the stuck-worker fallback, a snapshot
 * for the server every other second. */
let diagTick = 0;
setInterval(() => {
  if (DEBUG) renderDebug();
  if (!running) return;
  const now = performance.now();
  if (!stall && now - Math.max(diag.lastResultAt, diag.sessionAt) > 3000) {
    stall = { at: now, reason: stallReason(now) };
    emit("stall", { reason: stall.reason, ...snapshot(now) });
    setStatus("error", "Stalled", stall.reason);
  }
  // A threaded worker that stops reporting is stuck (a wasm thread that
  // crashed leaves the others waiting): fall back to one thread.
  if (workerReady?.threaded && !restartingWorker && now - workerBeatAt > 6000) fallBackToOneThread(now);
  if (++diagTick % 2 === 0) emit("stats", snapshot(now));
}, 1000);

async function fallBackToOneThread(now) {
  restartingWorker = true;
  emit("worker-stuck", { silentMs: Math.round(now - workerBeatAt), threads: workerReady.threads, ...snapshot(now) });
  flush();
  worker.terminate();
  workerReady = null;
  prefs.threads = false; // for this page load
  createWorker(false);
  toast("The multi-threaded VIO stopped responding: switching to one thread", 4000);
  try {
    await waitFor(() => workerReady, 15000, "the WebAssembly module did not load");
    if (running) beginSession("single-thread");
  } catch (err) {
    emit("error", { where: "page", message: text(err) });
  }
  restartingWorker = false;
}

function renderDebug() {
  const el = $("debug");
  el.hidden = false;
  const lines = Object.entries(snapshot()).map(([k, v]) => `${k} ${JSON.stringify(v)}`);
  el.textContent = [`session ${sessionId}${stall ? `  STALLED: ${stall.reason}` : ""}`, ...lines].join("\n");
}

$("video").addEventListener("resize", () => emit("video-resize", { width: $("video").videoWidth, height: $("video").videoHeight, running }));

// ---------------------------------------------------------------- recording

function startRecording() {
  recording = { frames: [], times: [], imu: [], started: performance.now() };
  $("b-rec").setAttribute("aria-pressed", "true");
  $("b-rec").textContent = "Stop rec";
  $("m-rec").hidden = false;
}

function finishRecording() {
  const r = recording;
  recording = null;
  $("b-rec").setAttribute("aria-pressed", "false");
  $("b-rec").textContent = "Record";
  $("m-rec").hidden = true;
  if (!r || r.frames.length < 10) return;
  // IMU samples must start before the first frame and be in time order.
  const imu = new Float64Array(r.imu);
  const header = {
    rig,
    width: frameW,
    height: frameH,
    frames: r.times,
    imu: imu.length / 7,
    device: { ...device, ua: navigator.userAgent, screen: [screen.width, screen.height, devicePixelRatio] },
    calibration: calib,
    accelSign,
    screenAngle,
    frameClock: FRAME_CLOCKS[frameClock],
    motionClock,
    telemetrySession: sessionId,
    imuRateHz,
    // Every camera frame is recorded; the page tracked every `trackedEvery`th.
    frameRateHz: Math.round((10 * (r.frames.length - 1)) / ((r.times[r.times.length - 1] - r.times[0]) / 1e9)) / 10,
    trackedEvery: prefs.fps60 && !tracking60() ? 2 : 1,
    video: stream?.getVideoTracks()[0]?.getSettings?.() ?? null,
    recordedAt: new Date().toISOString(),
  };
  lastRecording = recordingBlob(header, r.frames, [imu]);
  uploadState = null;
  const mb = (lastRecording.size / 1e6).toFixed(1);
  $("rec-info").textContent = `${r.frames.length} frames (${frameW}x${frameH}), ${imu.length / 7} motion samples, ${mb} MB.`;
  const a = $("b-download");
  if (a.href) URL.revokeObjectURL(a.href);
  a.href = URL.createObjectURL(lastRecording);
  a.download = `visloc-${device.key}-${Date.now()}.vrec`;
  $("up-progress").hidden = true;
  $("b-upload").disabled = false;
  $("b-upload").textContent = "Upload to the demo server";
  $("rec-sheet").hidden = false;
}

/** Uploads the last recording in chunks (a tunnel caps request bodies).
 * A chunk that fails is retried with a growing pause; the server answers a
 * chunk at the wrong offset with the length it has, and the upload resumes
 * there, so a retry after a failure continues instead of starting over. */
let uploadState = null; // { id, offset } of lastRecording
async function upload() {
  if (!lastRecording) return;
  uploadState ??= { id: `${new Date().toISOString().replace(/[:.]/g, "-")}-${device.key.replace(/[^a-zA-Z0-9-]/g, "_")}`, offset: 0 };
  const { id } = uploadState;
  const size = lastRecording.size;
  const chunk = 4 << 20;
  const bar = $("up-progress");
  bar.hidden = false;
  $("b-upload").disabled = true;
  $("b-upload").textContent = "Uploading…";
  let failures = 0;
  try {
    while (uploadState.offset < size) {
      const offset = uploadState.offset;
      const last = offset + chunk >= size ? 1 : 0;
      try {
        const res = await fetch(`upload?id=${encodeURIComponent(id)}&offset=${offset}&last=${last}`, { method: "POST", body: lastRecording.slice(offset, offset + chunk) });
        if (res.status === 409) {
          uploadState.offset = (await res.json()).have;
          continue;
        }
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        uploadState.offset = Math.min(size, offset + chunk);
        failures = 0;
        bar.firstElementChild.style.width = `${(100 * uploadState.offset) / size}%`;
      } catch (err) {
        if (++failures > 6) throw err;
        emit("upload-retry", { id, offset, failures, message: text(err) });
        $("b-upload").textContent = `Retrying (${failures})…`;
        await new Promise((resolve) => setTimeout(resolve, 1000 * failures));
      }
    }
    $("b-upload").textContent = `Uploaded: ${id}`;
    emit("upload", { id, bytes: size });
    toast("Uploaded. Thanks!");
  } catch (err) {
    $("b-upload").disabled = false;
    $("b-upload").textContent = `Upload stopped at ${Math.round((100 * uploadState.offset) / size)}%: tap to continue`;
    emit("error", { where: "upload", message: text(err), offset: uploadState.offset, bytes: size });
  }
}

// ---------------------------------------------------------------- settings

function fillSettings() {
  describeDevice($("device2"));
  $("i-focal").value = calib.focal35;
  $("i-td").value = calib.timeOffsetMs;
  $("i-noise").value = calib.noiseScale;
  $("i-width").value = String(prefs.width);
  $("i-sign").value = calib.accelSign === 1 ? "1" : calib.accelSign === -1 ? "-1" : "auto";
  $("i-gyro").value = calib.gyroAxes;
  $("i-watchdog").checked = prefs.watchdog;
  $("i-still").checked = prefs.stillStart;
  $("i-fps60").checked = prefs.fps60;
  $("i-track60").checked = prefs.track60;
  $("i-gpu").checked = prefs.gpu;
  $("i-threads").checked = prefs.threads;
  $("i-landmarks").checked = prefs.landmarks;
  $("i-tracks").checked = prefs.tracks;
  $("i-light").value = prefs.light;
  $("i-lead").value = prefs.overlayLeadMs;
  outputs();
  fillSyncInfo();
}
function outputs() {
  $("o-focal").textContent = `${$("i-focal").value} mm`;
  $("o-td").textContent = `${$("i-td").value} ms`;
  $("o-noise").textContent = `x${$("i-noise").value}`;
  $("o-lead").textContent = `${$("i-lead").value} ms`;
}
for (const id of ["i-focal", "i-td", "i-noise", "i-lead"]) $(id).addEventListener("input", outputs);
// The overlay timing applies at once, without a restart, so it can be
// tuned by eye.
$("i-lead").addEventListener("input", () => {
  prefs.overlayLeadMs = Number($("i-lead").value);
  localStorage.setItem("visloc-vio-overlay-lead", String(prefs.overlayLeadMs));
});

$("b-set").onclick = () => {
  fillSettings();
  $("settings-sheet").hidden = false;
};
$("b-close-set").onclick = () => ($("settings-sheet").hidden = true);
$("b-default").onclick = () => {
  clearCalibration(device);
  calib = defaultCalibration(device);
  fillSettings();
};
$("b-apply").onclick = () => {
  calib = {
    ...calib,
    focal35: Number($("i-focal").value),
    timeOffsetMs: Number($("i-td").value),
    noiseScale: Number($("i-noise").value),
    accelSign: $("i-sign").value === "auto" ? null : Number($("i-sign").value),
    gyroAxes: $("i-gyro").value,
  };
  saveCalibration(device, calib);
  const threadsChanged = prefs.threads !== $("i-threads").checked;
  prefs.width = Number($("i-width").value);
  prefs.watchdog = $("i-watchdog").checked;
  prefs.stillStart = $("i-still").checked;
  const fpsChanged = prefs.fps60 !== $("i-fps60").checked;
  prefs.fps60 = $("i-fps60").checked;
  prefs.track60 = $("i-track60").checked;
  prefs.gpu = $("i-gpu").checked;
  prefs.threads = $("i-threads").checked;
  prefs.landmarks = $("i-landmarks").checked;
  prefs.tracks = $("i-tracks").checked;
  prefs.light = $("i-light").value;
  localStorage.setItem("visloc-vio-width", String(prefs.width));
  localStorage.setItem("visloc-vio-watchdog", prefs.watchdog ? "1" : "0");
  localStorage.setItem("visloc-vio-still", prefs.stillStart ? "1" : "0");
  localStorage.setItem("visloc-vio-60fps:v2", prefs.fps60 ? "1" : "0");
  localStorage.setItem("visloc-vio-track60:v2", prefs.track60 ? "1" : "0");
  localStorage.setItem("visloc-vio-gpu", prefs.gpu ? "1" : "0");
  localStorage.setItem("visloc-vio-threads", prefs.threads ? "1" : "0");
  localStorage.setItem("visloc-vio-landmarks", prefs.landmarks ? "1" : "0");
  localStorage.setItem("visloc-vio-tracks", prefs.tracks ? "1" : "0");
  localStorage.setItem("visloc-vio-light", prefs.light);
  $("settings-sheet").hidden = true;
  if (threadsChanged) {
    location.reload();
    return;
  }
  const track = stream?.getVideoTracks()[0];
  if (fpsChanged && track) {
    track
      .applyConstraints(videoConstraints())
      .then(() => emit("camera", { label: track.label, settings: track.getSettings?.() ?? null, reason: "frame rate" }))
      .catch((err) => emit("error", { where: "camera", message: `frame rate: ${text(err)}` }));
  }
  if (running) beginSession("settings");
};

// ---------------------------------------------------------------- buttons

// A tap on the camera view moves the cube in front of the phone.
$("live").addEventListener("click", () => {
  if (running && imuOnly) {
    const c = imuOnlyCameraPose(lastFrameTime);
    placeCube({ camera: [...c.q, ...c.p] }, 1.2);
    mapDirty = true;
  } else if (running && latest && posed(latest.status)) {
    placeCube(latest, 1.2);
    mapDirty = true;
  }
});

$("b-start").onclick = start;
$("b-demo").onclick = startDemo;
$("b-stop").onclick = () => stop("button");
$("b-imu").onclick = () => (imuOnly ? stopImuOnly("button") : startImuOnly());
$("b-reset").onclick = () => {
  if (!running) return;
  stopImuOnly("reset");
  worker.postMessage({ type: "reset" });
  resetView();
  setStatus("init", ...startMessage());
};
$("b-rec").onclick = () => (recording ? finishRecording() : startRecording());
$("b-map").onclick = () => {
  mapVisible = !mapVisible;
  localStorage.setItem("visloc-vio-map", mapVisible ? "1" : "0");
  $("b-map").setAttribute("aria-pressed", String(mapVisible));
  $("map-wrap").hidden = !mapVisible;
  resizeAll();
};
$("b-map").setAttribute("aria-pressed", String(mapVisible));
$("b-upload").onclick = upload;
// A static host (GitHub Pages) takes neither uploads nor telemetry; the
// recording can still be downloaded.
hasServer.then((yes) => {
  serverHere = yes;
  $("b-upload").hidden = !yes;
});
$("b-close-rec").onclick = () => ($("rec-sheet").hidden = true);

boot();
