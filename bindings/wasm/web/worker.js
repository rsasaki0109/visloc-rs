// Owns the wasm VIO session. The page streams IMU samples and grey frames
// in; the worker always tracks the newest frame the IMU has caught up with
// (older ones are dropped: the IMU samples between them simply lengthen the
// next interval) and posts each result back.
//
// With `gpu` in the start message the session tracks points on the GPU
// (WebGPU), one frame ahead of the estimator (see gpuLoop).
//
// Messages in:  init { threads }, start { rig, overrides, watchdog, gpu,
//               estimateEvery },
//               imu { samples: Float64Array buffer }, frame { t, gray, width,
//               height }, reset, stop.
// Messages out: ready { threaded, threads, webgpu }, started { gpu: adapter
//               description or null }, result { ...update, ms, gpuWaitMs?,
//               landmarks? }, dropped { count }, error { message },
//               log { level, message }, stats { ... } (every second, also a
//               heartbeat: none arrive while the worker is stuck).

let wasm = null;
let vio = null;
let pending = null;
let dropped = 0;
let busy = false;
let framesSinceLandmarks = 0;
// The frame being tracked, awaited before the session is replaced or freed.
let inflight = null;
// While a session starts (the GPU device is requested), IMU samples wait
// here and no frame starts.
let starting = false;
let imuBacklog = [];
let deferredReset = false;

// Counters since the worker started, for the page's telemetry.
const stats = {
  imuMessages: 0,
  imuSamples: 0,
  imuAccepted: 0, // pushImu drops samples that are out of order or not finite
  imuFirstT: null,
  imuLastT: null, // newest sample time pushed (ns)
  frames: 0,
  replaced: 0, // frames dropped for a newer one before they were ready
  processed: 0,
  notReady: 0, // pump() found the pending frame ahead of the IMU
  errors: 0,
  lastMs: 0,
  maxMs: 0,
  lastImuSamples: null, // IMU samples the last processed frame consumed
  lastFrameT: null,
};

const text = (v) => (v instanceof Error ? `${v.name}: ${v.message}\n${v.stack ?? ""}` : String(v)).slice(0, 2000);

// The wasm panic hook prints to console.error; send that to the page.
for (const level of ["error", "warn"]) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    postMessage({ type: "log", level, message: args.map(text).join(" ") });
    original(...args);
  };
}
self.addEventListener("error", (e) => postMessage({ type: "error", message: `${e.message} (${e.filename}:${e.lineno})` }));
self.addEventListener("unhandledrejection", (e) => postMessage({ type: "error", message: text(e.reason) }));

setInterval(() => {
  let session = null;
  try {
    session = vio ? vio.debug() : null;
  } catch (err) {
    session = { error: text(err) };
  }
  postMessage({
    type: "stats",
    ...stats,
    busy,
    hasSession: !!vio,
    pendingT: pending?.t ?? null,
    session,
  });
}, 1000);

async function load(wantThreads) {
  const canThread = wantThreads && self.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined";
  if (canThread) {
    try {
      const mod = await import("./pkg-mt/visloc_wasm.js");
      await mod.default();
      const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
      await mod.initThreadPool(n);
      return mod;
    } catch (err) {
      postMessage({ type: "log", level: "warn", message: `threaded build failed (${text(err)}); using one thread` });
    }
  }
  const mod = await import("./pkg/visloc_wasm.js");
  await mod.default();
  return mod;
}

function pump() {
  if (busy || starting || !vio) return;
  if (vio.hasGpu()) {
    if (!gpuFrame && !pending) return;
    busy = true;
    inflight = gpuLoop(vio).finally(idle);
    return;
  }
  if (!pending) return;
  if (!vio.ready(pending.t)) {
    stats.notReady++;
    return; // the IMU has not reached this frame yet
  }
  const frame = pending;
  pending = null;
  busy = true;
  inflight = track(vio, frame).finally(idle);
}

function idle() {
  busy = false;
  inflight = null;
  if (deferredReset) {
    deferredReset = false;
    vio?.reset();
  }
  if (pending) setTimeout(pump, 0);
}

/** The CPU frontend: one frame, start to end. */
async function track(v, frame) {
  const started = performance.now();
  try {
    const u = v.process(frame.t, new Uint8Array(frame.gray));
    post(v, u, frame, { ms: performance.now() - started });
  } catch (err) {
    stats.errors++;
    postMessage({ type: "error", message: text(err), frameT: frame.t, ms: performance.now() - started });
  }
}

// The GPU frontend runs one frame ahead of the estimator: when a frame's
// tracks are back from the GPU, the next frame (if one is waiting) goes to
// the GPU before the estimator runs on the first, so the GPU's round trip
// overlaps the estimator instead of adding to it.
let gpuFrame = null; // { frame, done: Promise, started, submitMs, overlapped }

/** Sends the pending frame to the GPU, if there is one the IMU has reached. */
function submitPending(v, overlapped) {
  if (gpuFrame || !pending || starting) return;
  if (!v.ready(pending.t)) {
    stats.notReady++;
    return;
  }
  const frame = pending;
  pending = null;
  const started = performance.now();
  try {
    const done = v.submit(frame.t, new Uint8Array(frame.gray));
    gpuFrame = { frame, done, started, submitMs: performance.now() - started, overlapped };
  } catch (err) {
    stats.errors++;
    postMessage({ type: "error", message: text(err), frameT: frame.t, ms: 0 });
  }
}

async function gpuLoop(v) {
  while (v === vio && !starting) {
    if (deferredReset) {
      deferredReset = false;
      v.reset(); // cancels the frame on the GPU, if any
    }
    submitPending(v, false);
    const g = gpuFrame;
    if (!g) break;
    try {
      await g.done;
    } catch (err) {
      // A restart cancelled it.
      gpuFrame = null;
      postMessage({ type: "dropped", count: ++dropped });
      continue;
    }
    gpuFrame = null;
    if (v !== vio) break;
    const waitMs = performance.now() - g.started - g.submitMs;
    try {
      if (!v.takeGpu()) {
        postMessage({ type: "dropped", count: ++dropped });
        continue;
      }
      submitPending(v, true);
      const estimating = performance.now();
      const u = v.estimateGpu();
      if (u) {
        post(v, u, g.frame, {
          ms: performance.now() - g.started,
          gpuSubmitMs: g.submitMs,
          gpuWaitMs: waitMs,
          finishMs: performance.now() - estimating,
          overlapped: g.overlapped,
        });
      }
    } catch (err) {
      stats.errors++;
      postMessage({ type: "error", message: text(err), frameT: g.frame.t, ms: performance.now() - g.started });
    }
  }
}

/** Posts a frame's update to the page. */
function post(v, u, frame, timing) {
  stats.processed++;
  stats.lastMs = timing.ms;
  stats.maxMs = Math.max(stats.maxMs, timing.ms);
  stats.lastImuSamples = u.imuSamples;
  stats.lastFrameT = frame.t;
  const message = {
    type: "result",
    ...timing,
    frameId: u.frameId,
    t: u.t,
    camera: u.camera,
    imu: u.imu,
    velocity: u.velocity,
    accelBias: u.accelBias,
    gyroBias: u.gyroBias,
    keyframe: u.keyframe,
    trackIds: u.trackIds,
    trackXY: u.trackXY,
    connected: u.connected,
    imuSamples: u.imuSamples,
    status: u.status,
    resets: u.resets,
    width: frame.width,
    height: frame.height,
  };
  framesSinceLandmarks++;
  if (u.frameId % 30 === 29) message.sync = v.timeSync();
  if (u.keyframe || framesSinceLandmarks >= 15) {
    message.landmarks = v.landmarks();
    framesSinceLandmarks = 0;
  }
  const transfer = [message.trackIds.buffer, message.trackXY.buffer];
  if (message.landmarks) transfer.push(message.landmarks.buffer);
  postMessage(message, transfer);
}

/** Moves a GPU device into session `v`; its description, or null (the CPU
 * frontend stays). */
async function attachGpu(v) {
  if (!navigator.gpu) {
    postMessage({ type: "log", level: "warn", message: "no WebGPU in this worker: tracking on the CPU" });
    return null;
  }
  try {
    const gpu = await wasm.Gpu.create();
    const description = v.attachGpu(gpu);
    gpu.free();
    return description;
  } catch (err) {
    postMessage({ type: "log", level: "warn", message: `GPU frontend unavailable (${text(err)}): tracking on the CPU` });
    return null;
  }
}

onmessage = async (event) => {
  const m = event.data;
  try {
    switch (m.type) {
      case "init": {
        wasm = await load(m.threads);
        postMessage({ type: "ready", threaded: wasm.threadedBuild(), threads: wasm.threads(), webgpu: !!navigator.gpu });
        break;
      }
      case "start": {
        starting = true;
        if (inflight) await inflight;
        vio?.free();
        vio = null;
        pending = null;
        gpuFrame = null;
        dropped = 0;
        imuBacklog = [];
        try {
          const v = new wasm.Vio(m.rig, m.overrides);
          if (m.watchdog) v.setWatchdog(m.watchdog);
          if (m.estimateEvery > 1) v.setEstimateEvery(m.estimateEvery);
          const gpu = m.gpu ? await attachGpu(v) : null;
          vio = v;
          for (const samples of imuBacklog) stats.imuAccepted += vio.pushImu(samples);
          postMessage({ type: "started", gpu });
        } finally {
          imuBacklog = [];
          starting = false;
        }
        pump();
        break;
      }
      case "imu": {
        const samples = new Float64Array(m.samples);
        stats.imuMessages++;
        stats.imuSamples += samples.length / 7;
        if (samples.length >= 7) {
          stats.imuFirstT ??= samples[0];
          stats.imuLastT = samples[samples.length - 7];
        }
        if (starting) imuBacklog.push(samples);
        else if (vio) stats.imuAccepted += vio.pushImu(samples);
        pump();
        break;
      }
      case "frame": {
        stats.frames++;
        if (!vio && !starting) break;
        if (pending) {
          dropped++;
          stats.replaced++;
          postMessage({ type: "dropped", count: dropped });
        }
        pending = m;
        pump();
        break;
      }
      case "reset": {
        if (busy) deferredReset = true;
        else vio?.reset();
        gpuFrame = null;
        if (pending) postMessage({ type: "dropped", count: ++dropped });
        pending = null;
        break;
      }
      case "stop": {
        // The GPU loop ends once the session is no longer `vio`.
        const v = vio;
        vio = null;
        pending = null;
        if (inflight) await inflight;
        v?.free();
        break;
      }
    }
  } catch (err) {
    stats.errors++;
    postMessage({ type: "error", message: text(err), during: m.type });
  }
};
