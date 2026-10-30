// The .vrec recording format (see src/recording.rs): magic "VLOCREC2", a
// u32 header length, the JSON header, the IMU samples as f64s packed
// [t_ns, gx, gy, gz, ax, ay, az], then the grey frames (last, so a
// recording cut short keeps its whole frames). "VLOCREC1" files have the
// frames before the IMU samples.
const MAGIC = "VLOCREC2";

export function parseRecording(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const magic = new TextDecoder().decode(u8.subarray(0, 8));
  if (magic !== MAGIC && magic !== "VLOCREC1") throw new Error("not a .vrec recording");
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const len = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + len)));
  const frameSize = header.width * header.height;
  const imuBytes = header.imu * 56;
  const v1 = magic === "VLOCREC1";
  const imuStart = v1 ? 12 + len + header.frames.length * frameSize : 12 + len;
  const framesStart = v1 ? 12 + len : 12 + len + imuBytes;
  if (!v1) header.frames = header.frames.slice(0, Math.floor((u8.length - framesStart) / frameSize));
  const frames = u8.subarray(framesStart, framesStart + header.frames.length * frameSize);
  // Copied so the f64s are aligned (Uint8Array's slice: Node's
  // Buffer.slice would return a view).
  const imu = new Float64Array(Uint8Array.prototype.slice.call(u8, imuStart, imuStart + imuBytes).buffer);
  return { header, frames, imu };
}

/** Encodes a recording from the header object, frame byte chunks and IMU
 * Float64Array chunks, as a Blob (so a long capture is never one big
 * contiguous buffer). */
export function recordingBlob(header, frameChunks, imuChunks) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const head = new Uint8Array(12);
  head.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(head.buffer).setUint32(8, json.length, true);
  return new Blob([head, json, ...imuChunks, ...frameChunks], { type: "application/octet-stream" });
}
