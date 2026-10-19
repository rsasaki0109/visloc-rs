// Diagnostics for the demo server: the browser's features, the sensor
// clocks, how frames and motion samples flow to the VIO worker, its state,
// stalls and errors. Events are batched to `POST telemetry`, which serve.py
// appends to telemetry/<session>.jsonl. No images, no video. `?telemetry=0`
// turns it off.

export const enabled = new URLSearchParams(location.search).get("telemetry") !== "0";
export const sessionId = `${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}-${Math.random().toString(36).slice(2, 8)}`;

const MAX_QUEUED = 600;
const queue = [];
let batches = 0, lost = 0, failures = 0;

// serve.py marks its responses. A static host (GitHub Pages) has nowhere to
// send events: there they are dropped.
export const hasServer = fetch("./", { method: "HEAD", cache: "no-store" })
  .then((r) => r.headers.has("x-visloc-server"))
  .catch(() => false);
let server = null; // unknown until the probe answers; events wait
hasServer.then((yes) => {
  server = yes;
  if (!yes) queue.length = 0;
});

const repeats = new Map();

/** Queues an event; errors and stalls are sent at once. An error repeating
 * (say, every frame) is sent five times, then every hundredth time. */
export function emit(type, data = {}) {
  if (!enabled || server === false) return;
  if (type === "error" || type === "console") {
    const key = `${type}:${data.message}`.slice(0, 300);
    const count = (repeats.get(key) ?? 0) + 1;
    repeats.set(key, count);
    if (count > 5 && count % 100 !== 0) return;
    if (count > 1) data = { ...data, count };
  }
  queue.push({ type, t: Math.round(performance.now()), ...data });
  if (queue.length > MAX_QUEUED) lost += queue.splice(0, queue.length - MAX_QUEUED).length;
  if (type === "error" || type === "stall" || type === "start-failed") flush();
}

/** Sends the queued events (with `beacon`, in a way that survives the page
 * being closed). */
export function flush(beacon = false) {
  if (!enabled || !server || !queue.length) return;
  const events = queue.splice(0);
  const body = JSON.stringify({ session: sessionId, batch: batches++, lost, failures, events });
  if (beacon && navigator.sendBeacon?.("telemetry", body)) return;
  fetch("telemetry", { method: "POST", body, keepalive: body.length < 60000, headers: { "Content-Type": "application/json" } })
    .then((r) => {
      if (!r.ok) failures++;
    })
    .catch(() => {
      failures++;
      queue.unshift(...events.slice(-MAX_QUEUED / 2));
    });
}

/** A value as short text for an event (errors with their stack). */
export function text(v) {
  if (v instanceof Error) return `${v.name}: ${v.message}${v.stack ? `\n${v.stack}` : ""}`.slice(0, 2000);
  if (typeof v === "string") return v.slice(0, 2000);
  try {
    return JSON.stringify(v).slice(0, 2000);
  } catch {
    return String(v).slice(0, 2000);
  }
}

/** The browser and what it supports. */
export function environment() {
  return {
    url: location.href,
    ua: navigator.userAgent,
    platform: navigator.platform,
    screen: [screen.width, screen.height],
    dpr: devicePixelRatio,
    viewport: [innerWidth, innerHeight],
    cores: navigator.hardwareConcurrency ?? null,
    memoryGb: navigator.deviceMemory ?? null,
    isolated: self.crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    videoFrameCallback: "requestVideoFrameCallback" in HTMLVideoElement.prototype,
    motionPermission: typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function",
    timeOrigin: performance.timeOrigin,
    wall: Date.now(),
  };
}

if (enabled) {
  addEventListener("error", (e) => emit("error", { where: "page", message: e.message, source: `${e.filename}:${e.lineno}:${e.colno}`, stack: e.error?.stack?.slice(0, 2000) }));
  addEventListener("unhandledrejection", (e) => emit("error", { where: "page", message: text(e.reason) }));
  for (const level of ["error", "warn"]) {
    const original = console[level].bind(console);
    console[level] = (...args) => {
      emit("console", { level, message: args.map(text).join(" ").slice(0, 2000) });
      original(...args);
    };
  }
  setInterval(flush, 2000);
  addEventListener("pagehide", () => flush(true));
  document.addEventListener("visibilitychange", () => {
    emit("visibility", { hidden: document.hidden });
    if (document.hidden) flush(true);
  });
}
