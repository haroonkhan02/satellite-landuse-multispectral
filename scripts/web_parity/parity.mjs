// Run the browser demo's pipeline (web/satland.js + onnxruntime-web WASM) on test inputs dumped
// by scripts/web_parity.py and write what it produces next to them.
//
// Usage: node parity.mjs <dump dir> <models dir>

import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-web";

import { PATCH, TILE, classify, composite, detectChange, ndvi } from "../../web/satland.js";

const [dir, models] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
const load = async (name) => ort.InferenceSession.create(
  fs.readFileSync(path.join(models, name)), { executionProviders: ["wasm"] });
const timing = {};

// EuroSAT: class probabilities from both classifiers, plus the four display images.
const cubes = new Uint16Array(fs.readFileSync(path.join(dir, "eurosat.u16")).buffer.slice(0));
const per = 13 * PATCH * PATCH;
for (const name of ["all13", "rgb"]) {
  const session = await load(`eurosat_${name}_int8.onnx`);
  const out = new Float32Array(meta.eurosat * 10);
  const t0 = performance.now();
  for (let i = 0; i < meta.eurosat; i++) {
    out.set(await classify(session, ort, cubes.subarray(i * per, (i + 1) * per)), i * 10);
  }
  timing[name] = (performance.now() - t0) / meta.eurosat;
  fs.writeFileSync(path.join(dir, `js_${name}.f32`), out);
}
const views = [];
for (let i = 0; i < meta.views; i++) {
  const cube = cubes.subarray(i * per, (i + 1) * per);
  for (const kind of ["rgb", "cir", "swir"]) views.push(composite(cube, kind));
  views.push(ndvi(cube).rgba);
}
fs.writeFileSync(path.join(dir, "js_views.u8"), Buffer.concat(views.map((v) => Buffer.from(v))));

// LEVIR-CD: change probabilities from RGB tile pairs.
const tiles = fs.readFileSync(path.join(dir, "levir.u8"));
const session = await load("levircd_siamese_int8.onnx");
const n = TILE * TILE * 3;
const out = new Float32Array(meta.levir * TILE * TILE);
const t0 = performance.now();
for (let i = 0; i < meta.levir; i++) {
  const a = tiles.subarray(2 * i * n, (2 * i + 1) * n);
  const b = tiles.subarray((2 * i + 1) * n, (2 * i + 2) * n);
  out.set(await detectChange(session, ort, a, b, 3), i * TILE * TILE);
}
timing.change = (performance.now() - t0) / meta.levir;
fs.writeFileSync(path.join(dir, "js_change.f32"), out);
fs.writeFileSync(path.join(dir, "js_meta.json"), JSON.stringify({
  ort_version: ort.env.versions.web, ms_per_sample: timing,
}));
console.log(`node WASM ms per sample: ${JSON.stringify(timing)}`);
