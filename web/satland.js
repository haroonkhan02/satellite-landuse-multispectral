// Pre- and post-processing for the browser demo, ported from src/satland (bands.py, viz.py).
// Pure functions over typed arrays, so the same file runs in the page and in Node
// (scripts/web_parity/parity.mjs compares it with the Python pipeline).

export const BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10",
  "B11", "B12", "B8A"];
export const CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
  "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"];
export const PATCH = 64;
export const TILE = 256;
const COMBOS = { rgb: ["B04", "B03", "B02"], cir: ["B08", "B04", "B03"],
  swir: ["B12", "B08", "B04"] };

// numpy.percentile with the default linear interpolation, on a sorted copy.
export function percentile(values, q) {
  const s = Float64Array.from(values).sort();
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

function band(cube, name, n) {
  const k = BANDS.indexOf(name);
  return cube.subarray(k * n, (k + 1) * n);
}

// (13, H, W) digital numbers -> RGBA display image, like bands.composite: dark-object
// subtraction per channel (2nd percentile), then one joint stretch to the 98th percentile.
export function composite(cube, kind, n = PATCH * PATCH) {
  const chans = COMBOS[kind].map((b) => Float32Array.from(band(cube, b, n)));
  for (const c of chans) {
    const dark = Math.fround(percentile(c, 2));
    for (let i = 0; i < n; i++) c[i] = Math.max(0, Math.fround(c[i] - dark));
  }
  const all = new Float32Array(3 * n);
  chans.forEach((c, k) => all.set(c, k * n));
  const hi = Math.max(Math.fround(percentile(all, 98)), 1e-6);
  const out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = Math.min(1, Math.max(0, Math.fround(chans[k][i] / hi)));
      out[4 * i + k] = Math.floor(Math.fround(v * 255));
    }
    out[4 * i + 3] = 255;
  }
  return out;
}

// NDVI -> brown .. yellow .. green, like viz.ndvi_rgb. Returns {rgba, mean}.
export function ndvi(cube, n = PATCH * PATCH) {
  const red = band(cube, "B04", n);
  const nir = band(cube, "B08", n);
  const stops = [[120, 80, 50], [235, 220, 120], [30, 130, 50]];
  const out = new Uint8ClampedArray(4 * n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.min(1, Math.max(-1, (nir[i] - red[i]) / (nir[i] + red[i] + 1e-6)));
    sum += v;
    const t = Math.min(0.8, Math.max(-0.2, v)) + 0.2;
    const [a, b, f] = t < 0.5 ? [stops[0], stops[1], Math.min(1, t * 2)]
      : [stops[1], stops[2], Math.min(1, Math.max(0, t * 2 - 1))];
    for (let k = 0; k < 3; k++) out[4 * i + k] = Math.floor(a[k] + (b[k] - a[k]) * f);
    out[4 * i + 3] = 255;
  }
  return { rgba: out, mean: sum / n };
}

// Class probabilities for one (13, 64, 64) cube of raw digital numbers.
// The ONNX graph does the scaling, band selection, standardisation and softmax.
export async function classify(session, ort, cube) {
  const input = new ort.Tensor("float32", Float32Array.from(cube), [1, 13, PATCH, PATCH]);
  const out = await session.run({ bands: input });
  return Float32Array.from(out.probs.data);
}

export function topK(probs, k = 3) {
  return [...probs].map((p, i) => ({ label: CLASSES[i], p })).sort((a, b) => b.p - a.p)
    .slice(0, k);
}

// RGBA (or RGB with channels = 3) pixels of a 256x256 tile -> planar float32 in [0, 1].
export function planar(pixels, channels = 4, n = TILE * TILE) {
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) out[k * n + i] = pixels[channels * i + k] / 255;
  }
  return out;
}

// Change probability (256x256) for a before / after tile pair.
export async function detectChange(session, ort, before, after, channels = 4) {
  const shape = [1, 3, TILE, TILE];
  const out = await session.run({
    before: new ort.Tensor("float32", planar(before, channels), shape),
    after: new ort.Tensor("float32", planar(after, channels), shape),
  });
  return Float32Array.from(out.probs.data);
}

// Overlay on the dimmed 'after' tile, as viz.change_overlay: with a ground-truth mask green =
// correct, red = missed, blue = false alarm; without one, predicted change in orange.
export function changeOverlay(after, probs, truth = null, threshold = 0.5) {
  const n = probs.length;
  const out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const p = probs[i] > threshold;
    const t = truth ? truth[i] > 0 : false;
    let rgb = [after[4 * i] * 0.55, after[4 * i + 1] * 0.55, after[4 * i + 2] * 0.55];
    if (truth) {
      if (p && t) rgb = [40, 200, 70];
      else if (!p && t) rgb = [220, 50, 50];
      else if (p && !t) rgb = [60, 110, 240];
    } else if (p) rgb = [255, 140, 0];
    out.set([rgb[0], rgb[1], rgb[2], 255], 4 * i);
  }
  return out;
}
