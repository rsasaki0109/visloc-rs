// Replays a .vrec recording through the single-threaded wasm build under
// Node, for checking the browser build against the native replay
// (`cargo run --release --example replay -- <file.vrec>`) and timing it.
//
//   node web/node_replay.mjs <file.vrec> [--print-every 300]
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, { initSync, Vio } from "./pkg/visloc_wasm.js";
import { parseRecording } from "./recording.js";

const args = process.argv.slice(2);
const file = args[0];
const every = args.includes("--print-every") ? Number(args[args.indexOf("--print-every") + 1]) : 300;
initSync({ module: readFileSync(fileURLToPath(new URL("./pkg/visloc_wasm_bg.wasm", import.meta.url))) });

const rec = parseRecording(readFileSync(file));
const { header, frames, imu } = rec;
const n = header.width * header.height;
const vio = new Vio(header.rig, undefined);
let next = 0;
const times = [];
let last = null;
let path = 0;
let resets = 0;
const started = performance.now();
for (let i = 0; i < header.frames.length; i++) {
  const t = header.frames[i];
  while (next < header.imu && !vio.ready(t)) {
    vio.pushImu(imu.subarray(next * 7, next * 7 + 7));
    next++;
  }
  if (!vio.ready(t)) break;
  const tic = performance.now();
  const u = vio.process(t, frames.subarray(i * n, (i + 1) * n));
  times.push(performance.now() - tic);
  const p = [u.camera[4], u.camera[5], u.camera[6]];
  if (last) path += Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]);
  last = p;
  if (u.status.startsWith("reset")) resets++;
  if (every > 0 && i % every === 0) {
    console.log(`frame ${i} tracks ${u.trackIds.length} connected ${u.connected} ${u.status} p (${p.map((v) => v.toFixed(2)).join(", ")}) (${times.at(-1).toFixed(1)} ms)`);
  }
}
times.sort((a, b) => a - b);
const mean = times.reduce((a, b) => a + b, 0) / times.length;
console.log(
  `${times.length} frames in ${((performance.now() - started) / 1000).toFixed(1)} s, ${resets} resets; per frame mean ${mean.toFixed(1)} ms, p50 ${times[times.length >> 1].toFixed(1)}, p95 ${times[Math.floor(times.length * 0.95)].toFixed(1)}; path ${path.toFixed(2)} m; final p [${last}]`,
);
