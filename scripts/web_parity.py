"""Check the browser demo's JavaScript pipeline against the Python one on test data.

Dumps EuroSAT test patches (raw uint16) and decoded LEVIR-CD test tile pairs, runs
``scripts/web_parity/parity.mjs`` (web/satland.js + onnxruntime-web WASM, the same code the
static Space runs) and compares with ``OnnxRunner`` on the same INT8 files: label agreement,
largest probability difference and accuracy for both classifiers; flipped change pixels and
pooled F1 for the change model; and the composites / NDVI images against bands.composite and
viz.ndvi_rgb.

Example (after `npm install` in scripts/web_parity):
    python scripts/web_parity.py --eurosat 1000 --levir 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from satland.bands import composite
from satland.change import Confusion
from satland.config import OUTPUTS_DIR, REPORTS_DIR
from satland.eurosat import load_cache, load_split
from satland.export import OnnxRunner
from satland.export_models import EXPORT_DIR, cd_feed
from satland.levircd import decode_row, load_rows
from satland.viz import ndvi_rgb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eurosat", type=int, default=1000, help="test patches (random)")
    parser.add_argument("--levir", type=int, default=200, help="test tiles (random)")
    parser.add_argument("--views", type=int, default=100, help="patches for the image check")
    parser.add_argument("--models", type=Path, default=EXPORT_DIR)
    parser.add_argument("--work", type=Path, default=OUTPUTS_DIR / "web_parity")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    cache, files, labels = load_cache()
    idx = np.sort(rng.choice(load_split(files)["test"], args.eurosat, replace=False))
    cubes = np.asarray(cache[idx])
    rows = load_rows("test")
    rows = [rows[i] for i in np.sort(rng.choice(len(rows), args.levir, replace=False))]
    pairs = [decode_row(r) for r in rows]

    if args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True)
    cubes.astype("<u2").tofile(args.work / "eurosat.u16")
    np.stack([np.stack([a, b]) for a, b, _ in pairs]).astype(np.uint8).tofile(
        args.work / "levir.u8")
    (args.work / "meta.json").write_text(json.dumps(
        {"eurosat": len(idx), "levir": len(rows), "views": args.views}))
    subprocess.run(["node", str(ROOT / "scripts/web_parity/parity.mjs"), str(args.work),
                    str(args.models)], check=True)
    js_meta = json.loads((args.work / "js_meta.json").read_text())
    report = {"onnxruntime_web": js_meta["ort_version"],
              "node_wasm_ms_per_sample": js_meta["ms_per_sample"], "eurosat": {}, "levir": {}}

    y = labels[idx]
    feed = {"bands": cubes.astype(np.float32)}
    for name in ("all13", "rgb"):
        py = OnnxRunner(args.models / f"eurosat_{name}_int8.onnx").run(feed)
        js = np.fromfile(args.work / f"js_{name}.f32", np.float32).reshape(py.shape)
        report["eurosat"][name] = {
            "patches": len(idx), "label_agreement": float((py.argmax(1) == js.argmax(1)).mean()),
            "max_prob_diff": float(np.abs(py - js).max()),
            "accuracy_python": float((py.argmax(1) == y).mean()),
            "accuracy_js": float((js.argmax(1) == y).mean())}

    js_views = np.fromfile(args.work / "js_views.u8", np.uint8).reshape(args.views, 4, 64, 64, 4)
    diffs = {k: [] for k in ("rgb", "cir", "swir", "ndvi")}
    for i in range(args.views):
        cube = cubes[i]
        refs = [composite(cube, k) for k in ("rgb", "cir", "swir")] + [ndvi_rgb(cube)]
        for k, ref, js in zip(diffs, refs, js_views[i], strict=True):
            diffs[k].append(np.abs(ref.astype(int) - js[..., :3].astype(int)).max())
    report["eurosat"]["views_max_grey_level_diff"] = {k: int(max(v)) for k, v in diffs.items()}
    report["eurosat"]["views_identical_share"] = {
        k: float(np.mean(np.asarray(v) == 0)) for k, v in diffs.items()}

    run = OnnxRunner(args.models / "levircd_siamese_int8.onnx")
    js = np.fromfile(args.work / "js_change.f32", np.float32).reshape(len(rows), 256, 256)
    conf_py, conf_js, flips, max_diff = Confusion(), Confusion(), 0, 0.0
    for i, (_, _, m) in enumerate(pairs):
        py = run.run(cd_feed([rows[i]]))[0, 0]
        conf_py.update(py > 0.5, m)
        conf_js.update(js[i] > 0.5, m)
        flips += int(((py > 0.5) != (js[i] > 0.5)).sum())
        max_diff = max(max_diff, float(np.abs(py - js[i]).max()))
    report["levir"] = {"tiles": len(rows), "flipped_pixel_share": flips / js.size,
                       "max_prob_diff": max_diff, "f1_python": conf_py.f1(),
                       "f1_js": conf_js.f1()}
    (REPORTS_DIR / "web_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    shutil.rmtree(args.work)


if __name__ == "__main__":
    main()
