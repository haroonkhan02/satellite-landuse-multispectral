import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

import {
  CLASSES, PATCH, TILE, changeOverlay, classify, composite, detectChange, ndvi, topK,
} from "./satland.js";

// ?models=<url> loads the ONNX files from elsewhere (e.g. a local copy while developing).
const HUB = new URLSearchParams(location.search).get("models")
  || "https://huggingface.co/haroonx02/satellite-landuse-multispectral/resolve/main/onnx/";
const URLS = { all13: `${HUB}eurosat_all13_int8.onnx`, rgb: `${HUB}eurosat_rgb_int8.onnx`,
  change: `${HUB}levircd_siamese_int8.onnx` };
const GEOTIFF = "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";

const sessions = {};
let manifest = null;
const $ = (id) => document.getElementById(id);
const threads = () => `${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? "" : "s"}`;

async function fetchBytes(url, label, statusEl) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label} download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (statusEl) {
      statusEl.textContent = total ? `Downloading the ${label}… ${Math.round((100 * got) / total)} %`
        : `Downloading the ${label}… ${(got / 1e6).toFixed(1)} MB`;
    }
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

async function session(name, label, statusEl) {
  if (!sessions[name]) {
    sessions[name] = fetchBytes(URLS[name], label, statusEl)
      .then((m) => ort.InferenceSession.create(m, { executionProviders: ["wasm"] }));
  }
  return sessions[name];
}

function paint(canvasId, rgba, w, h) {
  const c = $(canvasId);
  c.width = w;
  c.height = h;
  c.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
}

function select(container, button) {
  container.querySelectorAll("button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b === button)));
}

// ------------------------------------------------------------------------- land use
function bars(el, probs, truth) {
  el.innerHTML = topK(probs, 3).map(({ label, p }) => {
    const mark = truth ? (label === truth ? " ✓" : "") : "";
    return `<div class="row"><span>${label}${mark}</span><div class="bar${
      label === topK(probs, 1)[0].label ? "" : " other"}" style="width:${(100 * p).toFixed(1)}%">
      </div><span>${(100 * p).toFixed(1)} %</span></div>`;
  }).join("");
}

async function showPatch(cube, truth, note) {
  for (const kind of ["rgb", "cir", "swir"]) paint(`v-${kind}`, composite(cube, kind), PATCH, PATCH);
  const nd = ndvi(cube);
  paint("v-ndvi", nd.rgba, PATCH, PATCH);
  const status = $("cls-status");
  const [all13, rgb] = await Promise.all([session("all13", "13-band model", status),
    session("rgb", "RGB model", status)]);
  const t0 = performance.now();
  const pa = await classify(all13, ort, cube);
  const pr = await classify(rgb, ort, cube);
  const ms = performance.now() - t0;
  bars($("p-all13"), pa, truth);
  bars($("p-rgb"), pr, truth);
  window.lastCls = { all13: pa, rgb: pr };
  status.textContent = `${truth ? `label: ${truth} · ` : ""}${note ? `${note} · ` : ""}mean NDVI `
    + `${nd.mean.toFixed(2)} · both models ${ms.toFixed(0)} ms (WASM, ${threads()})`;
}

async function loadExample(item, button) {
  select($("cls-examples"), button);
  try {
    const buf = await (await fetch(`examples/${item.file}`)).arrayBuffer();
    await showPatch(new Uint16Array(buf), item.label, item.note);
  } catch (err) {
    $("cls-status").textContent = String(err.message || err);
  }
}

async function readTif(file) {
  const { fromArrayBuffer } = await import(GEOTIFF);
  const image = await (await fromArrayBuffer(await file.arrayBuffer())).getImage();
  const w = image.getWidth();
  const h = image.getHeight();
  if (image.getSamplesPerPixel() !== 13) {
    throw new Error(`expected 13 Sentinel-2 bands, got ${image.getSamplesPerPixel()}`);
  }
  if (w < PATCH || h < PATCH) throw new Error(`image is ${w}×${h}; need at least 64×64`);
  const x0 = Math.floor((w - PATCH) / 2);
  const y0 = Math.floor((h - PATCH) / 2);
  const rasters = await image.readRasters({ window: [x0, y0, x0 + PATCH, y0 + PATCH],
    interleave: false });
  const cube = new Float32Array(13 * PATCH * PATCH);
  rasters.forEach((r, k) => cube.set(r, k * PATCH * PATCH));
  return { cube, note: w > PATCH || h > PATCH ? `centre 64×64 of ${w}×${h}` : "" };
}

// --------------------------------------------------------------------------- change
async function tilePixels(src, canvasId) {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`could not load ${src}`);
  const bitmap = await createImageBitmap(await res.blob(), { colorSpaceConversion: "none" });
  const c = $(canvasId);
  c.width = TILE;
  c.height = TILE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, TILE, TILE);
  return ctx.getImageData(0, 0, TILE, TILE).data;
}

async function showPair(beforeSrc, afterSrc, truthSrc) {
  const status = $("cd-status");
  try {
    const before = await tilePixels(beforeSrc, "c-before");
    const after = await tilePixels(afterSrc, "c-after");
    let truth = null;
    if (truthSrc) {
      const t = await tilePixels(truthSrc, "c-out");
      truth = new Uint8Array(TILE * TILE).map((_, i) => (t[4 * i] > 127 ? 1 : 0));
    }
    const model = await session("change", "change model (14 MB)", status);
    const t0 = performance.now();
    const probs = await detectChange(model, ort, before, after);
    const ms = performance.now() - t0;
    paint("c-out", changeOverlay(after, probs, truth), TILE, TILE);
    window.lastChange = probs;
    let changed = 0;
    for (const p of probs) changed += p > 0.5;
    $("cd-legend").innerHTML = truth
      ? '<span style="background:#28c846"></span>correct<span style="background:#dc3232"></span>'
        + 'missed<span style="background:#3c6ef0"></span>false alarm'
      : '<span style="background:#ff8c00"></span>predicted change';
    status.textContent = `${(100 * changed / probs.length).toFixed(1)} % of the tile changed · `
      + `${ms.toFixed(0)} ms (WASM, ${threads()})`;
  } catch (err) {
    status.textContent = String(err.message || err);
  }
}

// ---------------------------------------------------------------------------- setup
function buildExamples() {
  const strip = $("cls-examples");
  for (const item of manifest.eurosat) {
    const b = document.createElement("button");
    b.type = "button";
    b.title = item.note ? `${item.label} (${item.note})` : item.label;
    const c = document.createElement("canvas");
    c.width = PATCH;
    c.height = PATCH;
    b.append(c, item.label.replace("HerbaceousVegetation", "Herbaceous"));
    fetch(`examples/${item.file}`).then((r) => r.arrayBuffer()).then((buf) => {
      c.getContext("2d").putImageData(
        new ImageData(composite(new Uint16Array(buf), "rgb"), PATCH, PATCH), 0, 0);
    });
    b.addEventListener("click", () => loadExample(item, b));
    strip.appendChild(b);
  }
  const cd = $("cd-examples");
  for (const name of manifest.levir) {
    const b = document.createElement("button");
    b.type = "button";
    b.title = name;
    b.innerHTML = `<img src="examples/${name}_after.jpg" alt="${name}">${name.replace(/^\d_/, "").replace("_", " ")}`;
    b.addEventListener("click", () => {
      select(cd, b);
      showPair(`examples/${name}_before.jpg`, `examples/${name}_after.jpg`,
        `examples/${name}_label.png`);
    });
    cd.appendChild(b);
  }
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  select(b.parentElement, b);
  for (const id of ["landuse", "change"]) $(id).hidden = id !== b.dataset.tab;
  if (b.dataset.tab === "change" && !$("cd-status").textContent) $("cd-examples").firstChild.click();
}));

$("tif").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  select($("cls-examples"), null);
  try {
    const { cube, note } = await readTif(file);
    await showPatch(cube, null, note || file.name);
  } catch (err) {
    $("cls-status").textContent = `Could not read ${file.name}: ${err.message || err}`;
  }
});

const uploads = {};
for (const id of ["before", "after"]) {
  $(id).addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    uploads[id] = URL.createObjectURL(e.target.files[0]);
    select($("cd-examples"), null);
    if (uploads.before && uploads.after) showPair(uploads.before, uploads.after, null);
    else $("cd-status").textContent = `Now choose the ${id === "before" ? "after" : "before"} image.`;
  });
}

window.showLevir = (i) => $("cd-examples").children[i].click();
window.showEurosat = (i) => $("cls-examples").children[i].click();
window.CLASSES = CLASSES;

(async () => {
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  try {
    manifest = await (await fetch("examples/manifest.json")).json();
    buildExamples();
    await loadExample(manifest.eurosat[0], $("cls-examples").firstChild);
  } catch (err) {
    $("cls-status").textContent = `Could not start: ${err.message || err}`;
  }
})();
