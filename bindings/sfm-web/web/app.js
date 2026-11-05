// The page: photos (the samples in samples/samples.json, or the visitor's
// own) to worker.js, its progress on the bars, and the model in 3D (three.js,
// the browser VIO's bundle in vendor/): the points in their colours and each
// registered photo's camera.
import * as THREE from "three";
import { OrbitControls } from "three";

const $ = (id) => document.getElementById(id);
// `?cpu` keeps SIFT and matching off the GPU (to compare)
const SETTINGS = { maxSize: 800, maxKeypoints: 2000, window: 8, exhaustiveMax: 40,
  cpu: new URLSearchParams(location.search).has("cpu") };

// ---------------------------------------------------------------- 3D view
const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x0e1116);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
camera.position.set(4, 3, 4);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
let spin = true;
canvas.addEventListener("pointerdown", () => { spin = false; });
canvas.addEventListener("wheel", () => { spin = false; }, { passive: true });
let content = new THREE.Group();
scene.add(content);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
  }
}
let last = performance.now();
renderer.setAnimationLoop((now) => {
  resize();
  if (spin && content.children.length) {
    // turn about the scene's up through the target
    const dt = Math.min(0.1, (now - last) / 1000);
    const offset = camera.position.clone().sub(controls.target);
    offset.applyAxisAngle(camera.up, dt * 0.25);
    camera.position.copy(controls.target).add(offset);
  }
  last = now;
  controls.update();
  renderer.render(scene, camera);
});

function showModel(m) {
  scene.remove(content);
  content = new THREE.Group();
  scene.add(content);
  const n = m.points.length / 3;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(m.points, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(Array.from(m.colours, (c) => c / 255), 3));
  // robust centre and size: per-axis medians, 90th percentile distance
  const median = [0, 1, 2].map((a) => {
    const v = [];
    for (let i = a; i < m.points.length; i += 3) v.push(m.points[i]);
    v.sort((x, y) => x - y);
    return v[v.length >> 1] || 0;
  });
  const d = [];
  for (let i = 0; i < n; i++) d.push(Math.hypot(m.points[3 * i] - median[0], m.points[3 * i + 1] - median[1], m.points[3 * i + 2] - median[2]));
  d.sort((x, y) => x - y);
  const extent = d[Math.floor(0.9 * (d.length - 1))] || 1;
  content.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: extent / 250, vertexColors: true })));

  // cameras: a frustum per photo, world-to-camera rotation rows R, so a
  // pixel's ray is R^T [(u - cx) / fx, (v - cy) / fy, 1]
  const [fx, fy, cx, cy] = m.intrinsics;
  const s = extent * 0.04;
  const lines = [];
  let up = [0, 0, 0], forward = [0, 0, 0], mean = [0, 0, 0];
  for (const c of m.cameras) {
    const R = c.rotation;
    const ray = (u, v) => {
      const x = (u - cx) / fx, y = (v - cy) / fy;
      return [0, 1, 2].map((k) => c.centre[k] + s * (R[0][k] * x + R[1][k] * y + R[2][k]));
    };
    const corners = [ray(0, 0), ray(m.width, 0), ray(m.width, m.height), ray(0, m.height)];
    for (let k = 0; k < 4; k++) lines.push(c.centre, corners[k], corners[k], corners[(k + 1) % 4]);
    for (let k = 0; k < 3; k++) {
      up[k] -= R[1][k];
      forward[k] += R[2][k];
      mean[k] += c.centre[k] / m.cameras.length;
    }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.Float32BufferAttribute(lines.flat(), 3));
  content.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x35d0ba })));

  // the view: up as the photos', from behind the cameras towards the centre
  const u = new THREE.Vector3(...up).normalize();
  camera.up.copy(u);
  controls.target.set(...median);
  const back = new THREE.Vector3(...mean).sub(controls.target).add(u.clone().multiplyScalar(0.4 * extent));
  if (back.length() < 1e-6) back.set(...forward).multiplyScalar(-1);
  camera.position.copy(controls.target).add(back.normalize().multiplyScalar(2.2 * extent));
  camera.near = extent / 1000;
  camera.far = extent * 100;
  camera.updateProjectionMatrix();
  spin = true;
}

// ---------------------------------------------------------------- run
let worker = null;
// the last run: its photos (kept for the COLMAP export) and what the worker sent back
let photos = [];
let result = null;
let model = null;
const bar = (id, f) => { $(id).style.width = `${Math.round(100 * f)}%`; };
const log = (line) => { $("log").textContent += line + "\n"; $("log").scrollTop = 1e9; };
const busy = (on) => {
  $("sample").disabled = on;
  $("files").disabled = on;
  $("folder").disabled = on;
  document.body.classList.toggle("busy", on);
};

function reset() {
  for (const id of ["features", "match", "map"]) {
    bar(`b-${id}`, 0);
    $(`s-${id}`).textContent = "";
  }
  $("result").textContent = "";
  $("backend").textContent = "";
  $("downloads").hidden = true;
  $("log").textContent = "";
}

function run(list, focal) {
  reset();
  busy(true);
  photos = list;
  result = null;
  worker?.terminate();
  worker = new Worker("worker.js", { type: "module" });
  const n = photos.length;
  let tMatch = 0, start = null;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "backend") {
      $("backend").textContent = m.gpu !== null ? `SIFT and matching on the GPU (WebGPU: ${m.gpu}).`
        : `SIFT and matching on the CPU (${m.reason}).`;
      log(m.gpu !== null ? `WebGPU: ${m.gpu}` : `CPU: ${m.reason}`);
    } else if (m.type === "photo") {
      bar("b-features", (m.i + 1) / n);
      $("s-features").textContent = m.i + 1 < n ? `${m.i + 1} / ${n}` : `${n} / ${n}, ${(m.ms / 1000).toFixed(1)} s`;
      log(`${m.name}: ${m.keypoints} keypoints (${(m.ms / 1000).toFixed(1)} s)`);
    } else if (m.type === "focal") {
      start = m;
      log(m.from === "exif" ? `focal length from EXIF: ${m.mm35} mm (35 mm equivalent), ${m.px.toFixed(0)} px`
        : m.from === "given" ? `focal length given: ${m.px.toFixed(0)} px`
        : `no focal length in EXIF: starting from ${m.px.toFixed(0)} px (1.2 x the long side)`);
    } else if (m.type === "match") {
      bar("b-match", m.done / m.total);
      $("s-match").textContent = `${m.done} / ${m.total} pairs, ${m.verified} verified`;
      tMatch = m.ms;
    } else if (m.type === "mapping") {
      log(`matching: ${m.verified} pairs verified (${(tMatch / 1000).toFixed(1)} s)`);
      $("s-match").textContent += `, ${(tMatch / 1000).toFixed(1)} s`;
      $("s-map").textContent = "running…";
      bar("b-map", 0.5);
    } else if (m.type === "model") {
      result = m;
      model = JSON.parse(m.model);
      bar("b-map", 1);
      $("s-map").textContent = `${(m.ms / 1000).toFixed(1)} s`;
      const from = start?.from === "exif" ? ` (from ${start.px.toFixed(0)} px, EXIF)` : start?.from === "prior" ? ` (from ${start.px.toFixed(0)} px, a guess)` : "";
      $("result").innerHTML = `<b>${model.cameras.length} of ${n}</b> photos placed, <b>${(model.points.length / 3).toLocaleString()}</b> points, ` +
        `focal ${model.intrinsics[0].toFixed(1)} px${from} — ${(m.total_ms / 1000).toFixed(0)} s in all, on this device.`;
      $("downloads").hidden = false;
      showModel(model);
      busy(false);
    } else if (m.type === "error") {
      $("result").textContent = `Stopped: ${m.message}`;
      busy(false);
    }
  };
  // copied, not transferred: the page keeps the photos for the export
  worker.postMessage({ photos, focal, ...SETTINGS });
}

$("sample").addEventListener("click", async () => {
  const [s] = await (await fetch("samples/samples.json")).json();
  $("credit").innerHTML = s.credit;
  reset();
  $("s-features").textContent = "downloading…";
  const list = await Promise.all(s.files.map(async (f) => ({
    name: f,
    bytes: await (await fetch(`samples/${s.folder}/${f}`)).arrayBuffer(),
  })));
  run(list, s.focal);
});

// The visitor's photos: picked files, a picked folder or anything dropped
// on the page (files and folders), in name order.
const IMAGE = /\.(jpe?g|png|webp)$/i;
async function runFiles(files) {
  files = files.filter((f) => IMAGE.test(f.name))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  if (files.length < 2) {
    $("result").textContent = "Give it at least two photos (JPEG, PNG or WebP).";
    return;
  }
  $("credit").textContent = "";
  run(await Promise.all(files.map(async (f) => ({ name: f.name, bytes: await f.arrayBuffer() }))), 0);
}
$("files").addEventListener("change", (e) => runFiles([...e.target.files]));
$("folder").addEventListener("change", (e) => runFiles([...e.target.files]));

// every file under a dropped entry
async function entryFiles(entry) {
  if (entry.isFile) return [await new Promise((ok, fail) => entry.file(ok, fail))];
  if (!entry.isDirectory) return [];
  const reader = entry.createReader();
  const out = [];
  for (;;) {
    const batch = await new Promise((ok, fail) => reader.readEntries(ok, fail));
    if (!batch.length) return out;
    for (const e of batch) out.push(...(await entryFiles(e)));
  }
}
addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("drop");
});
addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("drop");
});
addEventListener("drop", async (e) => {
  e.preventDefault();
  document.body.classList.remove("drop");
  if (document.body.classList.contains("busy")) return;
  const entries = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  const files = entries.length ? (await Promise.all(entries.map(entryFiles))).flat() : [...e.dataTransfer.files];
  runFiles(files);
});

// ---------------------------------------------------------------- downloads
function save(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("dl-json").addEventListener("click", () => save("model.json", new Blob([JSON.stringify(model)], { type: "application/json" })));

function plyBlob() {
  const n = model.points.length / 3;
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\n` +
    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  const body = new DataView(new ArrayBuffer(n * 15));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) body.setFloat32(15 * i + 4 * k, model.points[3 * i + k], true);
    for (let k = 0; k < 3; k++) body.setUint8(15 * i + 12 + k, model.colours[3 * i + k]);
  }
  return new Blob([header, body.buffer], { type: "application/octet-stream" });
}
$("dl-ply").addEventListener("click", () => save("points.ply", plyBlob()));

// A zip of stored (uncompressed) entries: photos do not compress, and the
// text is small next to them.
const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function zip(files) {
  const parts = [], central = [];
  let offset = 0;
  for (const { name, data } of files) {
    const path = new TextEncoder().encode(name);
    const crc = crc32(data);
    const head = (sig, extra) => {
      const h = new DataView(new ArrayBuffer(extra));
      h.setUint32(0, sig, true);
      return h;
    };
    // local header: version 20, UTF-8 names (flag bit 11), stored
    const local = head(0x04034b50, 30);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, data.length, true);
    local.setUint32(22, data.length, true);
    local.setUint16(26, path.length, true);
    parts.push(local, path, data);
    const dir = head(0x02014b50, 46);
    dir.setUint16(4, 20, true);
    dir.setUint16(6, 20, true);
    dir.setUint16(8, 0x0800, true);
    dir.setUint32(16, crc, true);
    dir.setUint32(20, data.length, true);
    dir.setUint32(24, data.length, true);
    dir.setUint16(28, path.length, true);
    dir.setUint32(42, offset, true);
    central.push(dir, path);
    offset += 30 + path.length + data.length;
  }
  const size = central.reduce((s, p) => s + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, size, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], { type: "application/zip" });
}

// A COLMAP dataset as 3DGS trainers read it: images/ and sparse/0/ (text).
// A photo goes in as it is when its pixels are already upright, else
// redrawn upright (JPEG), as the browser showed it to the SfM.
$("dl-colmap").addEventListener("click", async () => {
  const button = $("dl-colmap");
  button.disabled = true;
  try {
    const text = (s) => new TextEncoder().encode(s);
    const [cameras, images, points] = result.colmap;
    const files = [
      { name: "sparse/0/cameras.txt", data: text(cameras) },
      { name: "sparse/0/images.txt", data: text(images) },
      { name: "sparse/0/points3D.txt", data: text(points) },
    ];
    for (const [k, c] of model.cameras.entries()) {
      button.textContent = `Packing ${k + 1} / ${model.cameras.length}…`;
      const bytes = photos[c.index].bytes;
      let data = new Uint8Array(bytes);
      if (result.orientations[c.index] > 1) {
        const bitmap = await createImageBitmap(new Blob([bytes]));
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        canvas.getContext("2d").drawImage(bitmap, 0, 0);
        bitmap.close();
        data = new Uint8Array(await (await canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 })).arrayBuffer());
      }
      files.push({ name: `images/${c.file}`, data });
    }
    save("colmap.zip", zip(files));
  } finally {
    button.textContent = "COLMAP (.zip)";
    button.disabled = false;
  }
});
