// Per-phone calibration: which phone this is, and the camera / IMU numbers
// the VIO needs for it.
//
// Android Chrome reports the exact model through User-Agent Client Hints.
// iOS hides the model, so an iPhone is identified by its screen (CSS size
// and pixel ratio), which narrows it to a family sharing a main camera.
//
// The intrinsics come from the main camera's 35 mm-equivalent focal length
// (the manufacturers' figure). getUserMedia streams cover the sensor's full
// width, so a 4:3 stream sees that focal length over its diagonal and a
// 16:9 one is that frame cropped top and bottom: either way
// f_px = f35 / 43.27 mm x 1.25 x (long side in pixels). Phone video pipelines
// correct most lens distortion; none is modelled unless a profile has it.
//
// `cameraOffset` is the camera's position relative to the IMU in the device
// frame (x right, y up the screen, z out of the screen), metres: the camera
// module sits near the top corner, the IMU near the middle of the board.
// The IMU noise densities are ADVIO's Kalibr values for an iPhone, x 3
// (what tracked best on its recordings through this pipeline at a
// browser-like 60 Hz), the same on every phone until measured otherwise.

export const IMU_DEFAULT = {
  accelNoise: 0.0144, // m/s²/√Hz
  gyroNoise: 0.0072, // rad/s/√Hz
  // Bias random walks 30x stiffer than the noise's ADVIO x3 scaling: a
  // looser bias lets the window absorb camera-gyro disagreements (blur,
  // fast pans) into the gyro bias, which on an iPhone ran to 15 mrad/s
  // where a bundle adjustment measured at most 2.6 (iOS corrects the gyro
  // bias itself). Stiffer: on 15 replays of three iPhone recordings, 3
  // watchdog resets instead of 9; on ADVIO's four walks, 3.76 m ATE
  // instead of 3.92.
  accelBiasWalk: 2.1e-5, // m/s³/√Hz
  gyroBiasWalk: 5.1e-6, // rad/s²/√Hz
};

const IPHONE_OFFSET = [0.02, 0.06, -0.005];
const ANDROID_OFFSET = [0.02, 0.06, -0.005];

// iPhones by screen (CSS px, portrait) and pixel ratio. Several models share
// a screen; `focal35` is the main camera of the most recent of them.
export const IPHONES = [
  { w: 440, h: 956, dpr: 3, label: "iPhone 16 Pro Max / 17 Pro Max", focal35: 24 },
  { w: 402, h: 874, dpr: 3, label: "iPhone 16 Pro / 17 / 17 Pro", focal35: 24 },
  { w: 420, h: 912, dpr: 3, label: "iPhone Air", focal35: 26 },
  { w: 430, h: 932, dpr: 3, label: "iPhone 15 Pro Max / 15 Plus / 16 Plus / 14 Pro Max", focal35: 24 },
  { w: 393, h: 852, dpr: 3, label: "iPhone 15 Pro / 15 / 16 / 16e / 14 Pro", focal35: 26 },
  { w: 428, h: 926, dpr: 3, label: "iPhone 14 Plus / 13 Pro Max / 12 Pro Max", focal35: 26 },
  { w: 390, h: 844, dpr: 3, label: "iPhone 14 / 13 / 13 Pro / 12 / 12 Pro", focal35: 26 },
  { w: 375, h: 812, dpr: 3, label: "iPhone 13 mini / 12 mini / 11 Pro / XS / X", focal35: 26 },
  { w: 414, h: 896, dpr: 3, label: "iPhone 11 Pro Max / XS Max", focal35: 26 },
  { w: 414, h: 896, dpr: 2, label: "iPhone 11 / XR", focal35: 26 },
  { w: 414, h: 736, dpr: 3, label: "iPhone 8 Plus / 7 Plus / 6s Plus", focal35: 28 },
  { w: 375, h: 667, dpr: 2, label: "iPhone SE (2nd/3rd) / 8 / 7 / 6s", focal35: 28 },
];

// Android phones by Client Hints model (a prefix match; Samsung reports
// codes such as SM-S921B).
export const ANDROIDS = [
  { model: /^Pixel 9 Pro/i, label: "Pixel 9 Pro / Pro XL", focal35: 24 },
  { model: /^Pixel 9/i, label: "Pixel 9", focal35: 24 },
  { model: /^Pixel 8 Pro/i, label: "Pixel 8 Pro", focal35: 24 },
  { model: /^Pixel 8/i, label: "Pixel 8 / 8a", focal35: 25 },
  { model: /^Pixel 7/i, label: "Pixel 7 / 7 Pro / 7a", focal35: 25 },
  { model: /^Pixel 6/i, label: "Pixel 6 / 6 Pro / 6a", focal35: 25 },
  { model: /^Pixel 10/i, label: "Pixel 10", focal35: 24 },
  { model: /^SM-S93/i, label: "Galaxy S25", focal35: 24 },
  { model: /^SM-S92/i, label: "Galaxy S24", focal35: 24 },
  { model: /^SM-S91/i, label: "Galaxy S23", focal35: 24 },
  { model: /^SM-S90/i, label: "Galaxy S22", focal35: 24 },
  { model: /^SM-G99/i, label: "Galaxy S21", focal35: 26 },
  // 108 MP main camera, 26 mm (the 13 mm ultra-wide is Chrome's default
  // back camera there: see preferMainCamera in app.js).
  { model: /^SM-N98[56]/i, label: "Galaxy Note20 Ultra", focal35: 26 },
  { model: /^SM-N98[01]/i, label: "Galaxy Note20", focal35: 26 },
  // 50 MP main camera, 24 mm (15 mm ultra-wide, 80 mm tele); Chrome
  // exposes one back camera. The online sync measured 0.92 x 26 mm.
  { model: /^A069P/i, label: "Nothing Phone (4a) Pro", focal35: 24 },
  { model: /^SM-A5/i, label: "Galaxy A5x", focal35: 26 },
  { model: /^CPH|^PH|^NE|^LE/i, label: "OnePlus / OPPO", focal35: 24 },
  { model: /^2[0-9]{3}[A-Z0-9]{4,}/i, label: "Xiaomi", focal35: 24 },
];

/** The phone's identity: `{ key, label, platform, model?, matched }`. */
export async function detectDevice() {
  const ua = navigator.userAgent;
  const ios = /iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  if (ios) {
    const w = Math.min(screen.width, screen.height);
    const h = Math.max(screen.width, screen.height);
    const dpr = Math.round(window.devicePixelRatio || 1);
    const hit = IPHONES.find((p) => p.w === w && p.h === h && p.dpr === dpr);
    return {
      key: `ios-${w}x${h}@${dpr}`,
      label: hit ? hit.label : `iPhone (${w}x${h} @${dpr}x)`,
      platform: "ios",
      focal35: hit?.focal35 ?? 26,
      cameraOffset: IPHONE_OFFSET,
      matched: Boolean(hit),
    };
  }
  let model = "";
  try {
    if (navigator.userAgentData?.getHighEntropyValues) {
      model = (await navigator.userAgentData.getHighEntropyValues(["model"])).model || "";
    }
  } catch {
    // Client Hints refused: fall back to the UA string.
  }
  if (!model) model = /Android [^;]+; ([^;)]+)/.exec(ua)?.[1]?.trim() ?? "";
  // Android UA strings now say "K" (Chrome) or "Mobile" (Firefox) instead
  // of the model; only Client Hints (Chromium) still give it.
  if (model === "K" || model === "Mobile") model = "";
  const android = /Android/.test(ua);
  const hit = ANDROIDS.find((p) => p.model.test(model));
  return {
    key: `${android ? "android" : "other"}-${model || "unknown"}`,
    label: hit ? `${hit.label} (${model})` : model || (android ? "Android phone" : "Desktop / unknown"),
    platform: android ? "android" : "other",
    browser: /Firefox\//.test(ua) ? "firefox" : "chromium",
    model,
    focal35: hit?.focal35 ?? 26,
    cameraOffset: ANDROID_OFFSET,
    matched: Boolean(hit),
  };
}

/** Camera timestamps relative to the motion events', ms, before the time
 * sync has measured a phone (it moves the offset by 15 ms or more). */
export const DEFAULT_TIME_OFFSET_MS = -25;

/** The default calibration for a device (before any per-device override). */
export function defaultCalibration(device) {
  return {
    focal35: device.focal35,
    // Camera timestamps relative to the motion events', ms (added to frame
    // times).
    timeOffsetMs: DEFAULT_TIME_OFFSET_MS,
    noiseScale: 1,
    cameraOffset: device.cameraOffset,
    // null: decide from the device orientation (see app.js).
    accelSign: null,
    // The device axes rotationRate's alpha / beta / gamma turn about:
    // "w3c" z / x / y (the spec; Firefox on Android), "webkit" x / y / z
    // (every iOS browser, measured on an iPhone 17-class phone with iOS 26;
    // and Chrome 155 on Android, measured on a Nothing Phone (4a) Pro, where
    // the camera check switched it within 12 s). Corrected from the camera
    // when it disagrees (app.js, checkGyroAxes).
    gyroAxes: device.platform === "android" && device.browser === "firefox" ? "w3c" : "webkit",
  };
}

/** Saved overrides for this device, merged over the defaults. A saved
 * focal length or time offset that is just the default it was saved under
 * (the page saves the whole calibration, e.g. when the time sync moves the
 * offset) gives way to a newer default; one measured or set by hand stays.
 * Saves from before `baseFocal35` / `baseTimeOffsetMs` were recorded had
 * the generic 26 mm and 0 ms. */
export function loadCalibration(device) {
  const base = defaultCalibration(device);
  try {
    const saved = JSON.parse(localStorage.getItem(`visloc-vio-calib:${device.key}`) || "{}");
    if (saved.focal35 != null && saved.focal35 === (saved.baseFocal35 ?? 26) && saved.focal35 !== device.focal35) {
      delete saved.focal35;
    }
    if (saved.timeOffsetMs != null && saved.timeOffsetMs === (saved.baseTimeOffsetMs ?? 0) && saved.timeOffsetMs !== DEFAULT_TIME_OFFSET_MS) {
      delete saved.timeOffsetMs;
    }
    delete saved.baseFocal35;
    delete saved.baseTimeOffsetMs;
    return { ...base, ...saved };
  } catch {
    return base;
  }
}

export function saveCalibration(device, calibration) {
  const saved = { ...calibration, baseFocal35: device.focal35, baseTimeOffsetMs: DEFAULT_TIME_OFFSET_MS };
  localStorage.setItem(`visloc-vio-calib:${device.key}`, JSON.stringify(saved));
}

export function clearCalibration(device) {
  localStorage.removeItem(`visloc-vio-calib:${device.key}`);
}

/** Pinhole intrinsics of a `width x height` frame from a 35 mm-equivalent
 * focal length (see the top of this file). */
export function intrinsics(width, height, focal35) {
  const f = (focal35 / 43.27) * 1.25 * Math.max(width, height);
  return { fx: f, fy: f, cx: (width - 1) / 2, cy: (height - 1) / 2 };
}
