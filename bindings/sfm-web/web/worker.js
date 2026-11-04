// The SfM run, off the page's thread: the wasm Session (pkg/, built by
// ../build.sh) stepped through features, matching and the mapper, with a
// message after each step. The browser decodes and shrinks each photo
// (applying its EXIF orientation); SIFT and matching run on WebGPU when the
// browser has it (and `cpu` is not set), else on the CPU.
//   in:  { photos: [{ name, bytes: ArrayBuffer }], maxSize, maxKeypoints, focal, window, exhaustiveMax, cpu }
//   out: { type: "backend", gpu (adapter label, or null), reason } | { type: "photo", i, name, keypoints, ms }
//        | { type: "match", done, total, verified, ms } | { type: "mapping", verified }
//        | { type: "model", model (JSON), ms, total_ms } | { type: "error", message }
import init, { Session } from "./pkg/visloc_sfm_web.js";

const ready = init();

// Pairs per GPU dispatch: a dispatch must end well inside the OS's GPU
// watchdog (Windows resets the GPU after 2 s), on integrated GPUs too.
const GPU_BATCH = 16;

// The photo's RGBA pixels, its long side shrunk to at most `maxSize`.
async function pixels(bytes, maxSize) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const s = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * s));
  const height = Math.max(1, Math.round(bitmap.height * s));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return { width, height, rgba: ctx.getImageData(0, 0, width, height).data };
}

// What the browser tells about the adapter (vendor and architecture).
async function adapterLabel() {
  const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  return a ? [a.info.vendor, a.info.architecture].filter(Boolean).join(" ") : "";
}

onmessage = async (e) => {
  await ready;
  const { photos, maxSize, maxKeypoints, focal, window, exhaustiveMax, cpu } = e.data;
  const session = new Session(maxKeypoints, focal);
  const t0 = performance.now();
  try {
    let gpu = null, reason = cpu ? "CPU chosen" : "";
    if (!cpu && self.navigator.gpu) {
      try {
        const name = await session.init_gpu();
        gpu = name || (await adapterLabel()) || "an adapter";
      } catch (err) {
        reason = String(err);
      }
    } else if (!cpu) {
      reason = "this browser has no WebGPU";
    }
    postMessage({ type: "backend", gpu, reason });
    for (let i = 0; i < photos.length; i++) {
      let image;
      try {
        image = await pixels(photos[i].bytes, maxSize);
      } catch {
        throw new Error(`${photos[i].name}: this browser cannot decode it`);
      }
      const keypoints = await session.add_photo(photos[i].name, image.width, image.height, image.rgba);
      postMessage({ type: "photo", i, name: photos[i].name, keypoints, ms: performance.now() - t0 });
    }
    const total = session.plan_pairs(window, exhaustiveMax);
    const tm = performance.now();
    // the CPU reports more often
    const batch = gpu !== null ? GPU_BATCH : 4;
    let left = total;
    while (left > 0) {
      left = await session.match_some(batch);
      postMessage({ type: "match", done: total - left, total, verified: session.verified_pairs(), ms: performance.now() - tm });
    }
    postMessage({ type: "mapping", verified: session.verified_pairs() });
    const tr = performance.now();
    const model = session.reconstruct();
    postMessage({ type: "model", model, ms: performance.now() - tr, total_ms: performance.now() - t0 });
  } catch (err) {
    postMessage({ type: "error", message: err instanceof Error ? err.message : String(err) });
  } finally {
    session.free();
  }
};
