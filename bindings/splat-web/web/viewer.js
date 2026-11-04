// The splat viewer page: the wasm renderer (pkg/, built by ../build.sh) on a
// full-window canvas, an orbit camera, and the scenes listed in scenes.json:
//   [{ "name", "file", "credit" (HTML), "cameras"?, "view"?, "up"?: [x, y, z], "pitch_deg"?, "distance"? }]
// "cameras" is the capture viewpoints gsplat_export_web --cameras writes (the
// world's up, the field of view and the first view come from them; ◀ ▶ step
// through them); without it the camera orbits the scene's centre, "up" up.
// `?scene=<index>` picks one, `?url=<file.splat>` shows any other. The view
// turns slowly until the first drag or zoom.
import init, { SplatViewer } from "./pkg/visloc_splat_web.js";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const MAX_PIXELS = 1.6e6; // the render size cap (a phone's full DPR is too slow)

function say(text) {
  $("message").textContent = text;
}

// ---------------------------------------------------------------- vectors
const add = (a, b) => a.map((v, i) => v + b[i]);
const scale = (a, s) => a.map((v) => v * s);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => scale(a, 1 / Math.hypot(...a));

// ---------------------------------------------------------------- orbit camera
const cam = { target: [0, 0, 0], up: [0, 0, 1], e1: [1, 0, 0], e2: [0, 1, 0], yaw: 0.6, pitch: 0.35, distance: 3, fov: (55 * Math.PI) / 180 };

function setUp(up) {
  cam.up = norm(up);
  const a = Math.abs(cam.up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  cam.e1 = norm(add(a, scale(cam.up, -dot(a, cam.up))));
  cam.e2 = cross(cam.up, cam.e1);
}

function eye() {
  const horiz = add(scale(cam.e1, Math.cos(cam.yaw)), scale(cam.e2, Math.sin(cam.yaw)));
  const dir = add(scale(horiz, Math.cos(cam.pitch)), scale(cam.up, Math.sin(cam.pitch)));
  return add(cam.target, scale(dir, cam.distance));
}

// Place the orbit camera at `from`, looking at `at`.
function lookFrom(from, at) {
  cam.target = at;
  const rel = add(from, scale(at, -1));
  cam.distance = Math.max(1e-3, Math.hypot(...rel));
  const dir = scale(rel, 1 / cam.distance);
  cam.pitch = Math.asin(Math.max(-0.99, Math.min(0.99, dot(dir, cam.up))));
  cam.yaw = Math.atan2(dot(dir, cam.e2), dot(dir, cam.e1));
  dirty = true;
}

let spin = true; // until the first drag or zoom
let dirty = true;
function orbit(dx, dy) {
  cam.yaw -= dx * 0.006;
  cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.006));
  dirty = true;
}
function pan(dx, dy) {
  const forward = norm(add(cam.target, scale(eye(), -1)));
  const right = norm(cross(forward, cam.up));
  const upv = cross(right, forward);
  const k = (cam.distance * Math.tan(cam.fov / 2) * 2) / canvas.clientHeight;
  cam.target = add(cam.target, add(scale(right, -dx * k), scale(upv, dy * k)));
  dirty = true;
}
function zoom(factor) {
  cam.distance = Math.max(1e-3, cam.distance * factor);
  dirty = true;
}

// mouse and touch
const pointers = new Map();
let pinch = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  stopTour();
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  if (pointers.size === 1) {
    if (p.button === 2 || p.button === 1 || p.shift || e.shiftKey) pan(dx, dy);
    else orbit(dx, dy);
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const before = Math.hypot(a.x - b.x, a.y - b.y);
    p.x = e.clientX; p.y = e.clientY;
    const after = Math.hypot(a.x - b.x, a.y - b.y);
    if (pinch !== null && before > 0) zoom(before / Math.max(after, 1));
    pan(dx / 2, dy / 2);
    pinch = after;
    return;
  }
  p.x = e.clientX; p.y = e.clientY;
});
const release = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
  if (pointers.size === 0) canvas.classList.remove("dragging");
};
canvas.addEventListener("pointerup", release);
canvas.addEventListener("pointercancel", release);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  stopTour();
  zoom(Math.exp(e.deltaY * 0.0012));
}, { passive: false });

// ---------------------------------------------------------------- scenes
async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value.subarray(0, Math.min(value.length, total - got)), got);
    got += value.length;
    $("progress").style.width = `${(100 * Math.min(got, total)) / total}%`;
  }
  return got === total ? out : out.subarray(0, got);
}

let viewer = null;
let busy = false;
let pending = null; // a scene load or resize waiting for the frame in flight

function whenIdle(task) {
  pending = task;
  dirty = true;
}

// The capture viewpoints of the scene shown (gsplat_export_web's cameras.json).
let views = [], viewIndex = 0, centre = [0, 0, 0], radius = 1;

function showView(i) {
  if (!views.length) return;
  viewIndex = (i + views.length) % views.length;
  const v = views[viewIndex];
  const [from, at] = viewPose(viewIndex);
  lookFrom(from, at);
  $("view-name").textContent = `capture view ${viewIndex + 1} / ${views.length}`;
}

// Capture view i as an eye and a look-at point: along the camera's axis, at
// the depth of the scene's centre.
function viewPose(i) {
  const v = views[(i + views.length) % views.length];
  const forward = v.rotation[2];
  const depth = Math.max(0.3 * radius, dot(add(centre, scale(v.center, -1)), forward));
  return [v.center, add(v.center, scale(forward, depth))];
}

// Until the first drag or zoom, glide from capture view to capture view (a
// splat from photos looks right near where they were taken, not around it).
const TOUR_SECONDS = 5; // per view
let tour = 0;
function stepTour(dt) {
  tour += dt / TOUR_SECONDS;
  const i = Math.floor(tour), f = tour - i;
  const s = f * f * f * (f * (6 * f - 15) + 10); // ease in and out: a pause at each view
  const [ea, ta] = viewPose(viewIndex + i), [eb, tb] = viewPose(viewIndex + i + 1);
  const mix = (a, b) => add(scale(a, 1 - s), scale(b, s));
  lookFrom(mix(ea, eb), mix(ta, tb));
  $("view-name").textContent = `capture view ${((viewIndex + i + Math.round(s)) % views.length) + 1} / ${views.length}`;
}
// The view the tour has reached, then a step from it.
const stopTour = () => {
  if (spin && views.length > 1) viewIndex = (viewIndex + Math.round(tour)) % views.length;
  spin = false;
  tour = 0;
};
$("prev").addEventListener("click", () => { stopTour(); showView(viewIndex - 1); });
$("next").addEventListener("click", () => { stopTour(); showView(viewIndex + 1); });

async function loadScene(entry) {
  say("");
  $("credit").innerHTML = entry.credit || "";
  $("stats").textContent = `loading ${entry.file}…`;
  $("progress").style.width = "0";
  const rig = entry.cameras ? await (await fetch(entry.cameras)).json() : null;
  const bytes = await fetchWithProgress(entry.file);
  $("progress").style.width = "0";
  whenIdle(() => {
    const [cx, cy, cz, r] = viewer.load_splat(bytes);
    centre = [cx, cy, cz];
    radius = r;
    views = rig?.views || [];
    tour = 0;
    $("views").hidden = views.length === 0;
    if (views.length) {
      // the world's up: against the capture cameras' mean image-down axis
      let down = [0, 0, 0];
      for (const v of views) down = add(down, v.rotation[1]);
      setUp(scale(down, -1));
      if (rig.fy && rig.height) cam.fov = Math.min(1.6, Math.max(0.6, 2 * Math.atan(rig.height / 2 / rig.fy)));
      showView(entry.view ?? 0);
    } else {
      setUp(entry.up || [0, 0, 1]);
      cam.target = centre;
      cam.distance = (entry.distance ?? 1.6) * radius;
      cam.yaw = entry.yaw_deg !== undefined ? (entry.yaw_deg * Math.PI) / 180 : 0.6;
      cam.pitch = ((entry.pitch_deg ?? 20) * Math.PI) / 180;
    }
  });
}

function fitCanvas() {
  const dpr = window.devicePixelRatio || 1;
  let w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  const s = Math.min(1, Math.sqrt(MAX_PIXELS / (w * h)));
  w = Math.max(1, Math.round(w * s));
  h = Math.max(1, Math.round(h * s));
  if (w !== canvas.width || h !== canvas.height) {
    canvas.width = w;
    canvas.height = h;
    if (viewer) {
      const prev = pending;
      whenIdle(() => {
        if (prev) prev();
        viewer.resize(w, h);
      });
    }
  }
}

// ---------------------------------------------------------------- frame loop
let frames = 0, lastFps = performance.now(), fps = 0, lastFrame = performance.now();
async function frame(now) {
  requestAnimationFrame(frame);
  if (!viewer || busy) return;
  fitCanvas();
  if (spin && viewer.num_gaussians() > 0) {
    const dt = Math.min(0.1, (now - lastFrame) / 1000);
    if (views.length > 1) stepTour(dt);
    else cam.yaw += dt * 0.12;
    dirty = true;
  }
  lastFrame = now;
  if (pending) {
    const task = pending;
    pending = null;
    try {
      task();
    } catch (err) {
      say(`Could not show the scene: ${err}`);
      return;
    }
  }
  if (!dirty) return;
  dirty = false;
  busy = true;
  try {
    await viewer.render(eye(), cam.target, cam.up, cam.fov);
  } catch (err) {
    say(`Rendering failed: ${err}`);
  }
  busy = false;
  frames++;
  const t = performance.now();
  if (t - lastFps > 500) {
    fps = (1000 * frames) / (t - lastFps);
    frames = 0;
    lastFps = t;
  }
  $("stats").textContent = `${viewer.num_gaussians().toLocaleString()} gaussians · ${canvas.width}×${canvas.height} · ${fps.toFixed(0)} fps · ${viewer.adapter()}`;
}

// ---------------------------------------------------------------- start
async function main() {
  if (!navigator.gpu) {
    say("This viewer needs WebGPU: a current Chrome, Edge or Safari (26 or later).");
    $("stats").textContent = "no WebGPU";
    return;
  }
  await init();
  fitCanvas();
  try {
    viewer = await SplatViewer.create(canvas);
  } catch (err) {
    say(`WebGPU did not start: ${err}`);
    return;
  }
  const params = new URLSearchParams(location.search);
  let scenes = [];
  try {
    scenes = await (await fetch("scenes.json")).json();
  } catch {}
  if (params.get("url")) scenes.unshift({ name: params.get("url"), file: params.get("url"), up: [0, 0, 1] });
  const select = $("scene");
  scenes.forEach((s, i) => select.add(new Option(s.name, i)));
  select.hidden = scenes.length < 2;
  select.addEventListener("change", () => loadScene(scenes[+select.value]).catch((e) => say(String(e))));
  const first = Math.min(scenes.length - 1, Math.max(0, Number(params.get("scene")) || 0));
  if (!scenes.length) {
    say("No scenes listed (scenes.json).");
    return;
  }
  select.value = first;
  requestAnimationFrame(frame);
  await loadScene(scenes[first]);
}

main().catch((err) => say(String(err)));
