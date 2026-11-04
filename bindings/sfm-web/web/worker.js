// The SfM run, off the page's thread: the wasm Session (pkg/, built by
// ../build.sh) stepped through features, matching and the mapper, with a
// message after each step.
//   in:  { photos: [{ name, bytes: ArrayBuffer }], maxSize, maxKeypoints, focal, window, exhaustiveMax }
//   out: { type: "photo", i, name, keypoints, ms } | { type: "match", done, total, verified, ms }
//        | { type: "mapping", verified } | { type: "model", model (JSON), ms } | { type: "error", message }
import init, { Session } from "./pkg/visloc_sfm_web.js";

const ready = init();

onmessage = async (e) => {
  await ready;
  const { photos, maxSize, maxKeypoints, focal, window, exhaustiveMax } = e.data;
  const session = new Session(maxSize, maxKeypoints, focal);
  const t0 = performance.now();
  try {
    for (let i = 0; i < photos.length; i++) {
      const keypoints = session.add_photo(photos[i].name, new Uint8Array(photos[i].bytes));
      postMessage({ type: "photo", i, name: photos[i].name, keypoints, ms: performance.now() - t0 });
    }
    const total = session.plan_pairs(window, exhaustiveMax);
    const tm = performance.now();
    let left = total;
    while (left > 0) {
      left = session.match_some(4);
      postMessage({ type: "match", done: total - left, total, verified: session.verified_pairs(), ms: performance.now() - tm });
    }
    postMessage({ type: "mapping", verified: session.verified_pairs() });
    const tr = performance.now();
    const model = session.reconstruct();
    postMessage({ type: "model", model, ms: performance.now() - tr, total_ms: performance.now() - t0 });
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  } finally {
    session.free();
  }
};
