"""Export both models and re-score every exported file on the full test sets.

For each model (EuroSAT all-13 and RGB classifiers, LEVIR-CD Siamese U-Net) this writes ONNX
FP32 / FP16 / INT8 and CoreML FP32 / FP16 files to models/onnx/, then runs the 5,400 EuroSAT
test patches or the 2,048 LEVIR-CD test tiles through each one and compares them with PyTorch:
test metrics, agreement of the predicted labels / change pixels, and the largest probability
difference. INT8 is calibrated on random training inputs (never test data); the classifiers use
percentile (99.999) ranges, because a few extreme pixels stretch MinMax ranges.

Writes reports/export.md and reports/export.json.

Example:
    satland-export                      # everything, ~15 min on an M5
    satland-export --models levircd --skip-coreml
"""

from __future__ import annotations

import argparse
import json
import platform
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .change import Confusion
from .config import MODELS_DIR, REPORTS_DIR
from .eurosat import load_cache, load_split
from .export import (
    CoreMLRunner,
    OnnxRunner,
    TorchRunner,
    export_coreml,
    export_onnx,
    load_change,
    load_classifier,
    to_fp16,
    to_int8,
)
from .levircd import decode_row, load_rows
from .models import pick_device

EXPORT_DIR = MODELS_DIR / "onnx"
MODELS = {"eurosat_all13": ("cls", MODELS_DIR / "eurosat_all13.pt"),
          "eurosat_rgb": ("cls", MODELS_DIR / "eurosat_rgb.pt"),
          "levircd_siamese": ("cd", MODELS_DIR / "levircd_absdiff.pt")}
VARIANTS = ("onnx_fp32", "onnx_fp16", "onnx_int8", "coreml_fp32", "coreml_fp16")


def file_name(model: str, variant: str) -> str:
    kind, precision = variant.split("_")
    return f"{model}_{precision}.onnx" if kind == "onnx" else f"{model}_{precision}.mlpackage"


def file_mb(path: Path) -> float:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return path.stat().st_size / 1e6


# ------------------------------------------------------------------------------- inputs
def cls_batches(cache, idx: np.ndarray, batch: int = 256):
    for s in range(0, len(idx), batch):
        yield {"bands": np.asarray(cache[np.sort(idx[s:s + batch])]).astype(np.float32)}


def cd_feed(rows) -> dict[str, np.ndarray]:
    pairs = [decode_row(r) for r in rows]
    a, b = (np.stack([p[k] for p in pairs]).transpose(0, 3, 1, 2).astype(np.float32) / 255
            for k in (0, 1))
    return {"before": np.ascontiguousarray(a), "after": np.ascontiguousarray(b)}


def cd_batches(rows, batch: int = 32):
    for s in range(0, len(rows), batch):
        yield cd_feed(rows[s:s + batch])


def calibration(kind: str, n: int, seed: int = 0) -> list[dict[str, np.ndarray]]:
    """Random training inputs, one per feed (batch 1, like the browser)."""
    rng = np.random.default_rng(seed)
    if kind == "cls":
        cache, files, _ = load_cache()
        idx = rng.choice(load_split(files)["train"], n, replace=False)
        return [{"bands": np.asarray(cache[i:i + 1]).astype(np.float32)} for i in idx]
    rows = load_rows("train")
    return [cd_feed([rows[i]]) for i in rng.choice(len(rows), n, replace=False)]


# ------------------------------------------------------------------------------ scoring
def score_cls(runner, ref: np.ndarray | None, cache, idx, labels) -> tuple[dict, np.ndarray]:
    probs, ms = [], 0.0
    for feed in cls_batches(cache, idx):
        out, t = runner.run_batches([feed])
        probs.append(out)
        ms += t * len(out)
    probs = np.concatenate(probs)
    pred, y = probs.argmax(1), labels[np.sort(idx)]
    res = {"accuracy": float((pred == y).mean()),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "errors": int((pred != y).sum()), "ms_per_sample": ms / len(probs)}
    if ref is not None:
        res["label_agreement"] = float((pred == ref.argmax(1)).mean())
        res["max_prob_diff"] = float(np.abs(probs - ref).max())
    return res, probs


def score_cd(runner, ref: np.ndarray | None, rows, batch: int = 32) -> tuple[dict, np.ndarray]:
    conf, flips, max_diff, ms, outs = Confusion(), 0, 0.0, 0.0, []
    for s, feed in zip(range(0, len(rows), batch), cd_batches(rows, batch), strict=True):
        out, t = runner.run_batches([feed])
        out = out[:, 0]
        ms += t * len(out)
        target = np.stack([decode_row(r)[2] for r in rows[s:s + batch]])
        conf.update(out > 0.5, target)
        if ref is not None:
            r = ref[s:s + len(out)].astype(np.float32)
            flips += int(((out > 0.5) != (r > 0.5)).sum())
            max_diff = max(max_diff, float(np.abs(out - r).max()))
        else:
            outs.append(out.astype(np.float16))
    res = {**conf.as_dict(), "ms_per_sample": ms / len(rows)}
    if ref is not None:
        res["flipped_pixel_share"] = flips / ref.size
        res["max_prob_diff"] = max_diff
    return res, (np.concatenate(outs) if outs else None)


# ------------------------------------------------------------------------------- export
def export_all(name: str, kind: str, checkpoint: Path, out_dir: Path, calib_n: int,
               calib_method: str, skip_coreml: bool) -> dict[str, Path]:
    wrapper = load_classifier(checkpoint)[0] if kind == "cls" else load_change(checkpoint)
    paths = {v: out_dir / file_name(name, v) for v in VARIANTS}
    export_onnx(wrapper, kind, paths["onnx_fp32"])
    to_fp16(paths["onnx_fp32"], paths["onnx_fp16"])
    to_int8(paths["onnx_fp32"], paths["onnx_int8"], calibration(kind, calib_n), calib_method)
    if skip_coreml:
        return {k: v for k, v in paths.items() if k.startswith("onnx")}
    for precision in ("fp32", "fp16"):
        export_coreml(wrapper, kind, paths[f"coreml_{precision}"], fp16=precision == "fp16")
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out", type=Path, default=EXPORT_DIR)
    ap.add_argument("--calib-cls", type=int, default=256, help="calibration patches (EuroSAT)")
    ap.add_argument("--calib-cd", type=int, default=64, help="calibration tiles (LEVIR-CD)")
    ap.add_argument("--method-cls", default="Percentile",
                    help="INT8 calibration for the classifiers (MinMax, Percentile, Entropy)")
    ap.add_argument("--method-cd", default="MinMax", help="INT8 calibration for LEVIR-CD")
    ap.add_argument("--skip-coreml", action="store_true")
    ap.add_argument("--device", default=None, help="PyTorch reference device")
    args = ap.parse_args(argv)
    if platform.system() != "Darwin":
        args.skip_coreml = True
    device = str(pick_device(args.device))
    import onnxruntime as ort
    import torch

    report_path = REPORTS_DIR / "export.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"onnxruntime": ort.__version__, "torch": torch.__version__,
                   "reference_device": device})
    models = report.setdefault("models", {})
    cache = files = labels = idx = rows = None
    for name in args.models:
        kind, ckpt = MODELS[name]
        print(f"== {name}: exporting", flush=True)
        cls = kind == "cls"
        paths = export_all(name, kind, ckpt, args.out, args.calib_cls if cls else args.calib_cd,
                           args.method_cls if cls else args.method_cd, args.skip_coreml)
        wrapper = load_classifier(ckpt)[0] if kind == "cls" else load_change(ckpt)
        if kind == "cls":
            if cache is None:
                cache, files, labels = load_cache()
                idx = load_split(files)["test"]
            score = partial(score_cls, cache=cache, idx=idx, labels=labels)
        else:
            rows = rows or load_rows("test")
            score = partial(score_cd, rows=rows)
        res, ref = score(TorchRunner(wrapper, device), None)
        entry = {"test_samples": len(idx) if kind == "cls" else len(rows),
                 "pytorch": {**res, "size_mb": file_mb(ckpt)}}
        print(f"   pytorch ({device}): {res}", flush=True)
        for variant, path in paths.items():
            runner = OnnxRunner(path) if variant.startswith("onnx") else CoreMLRunner(path)
            res, _ = score(runner, ref)
            entry[variant] = {**res, "size_mb": file_mb(path), "file": path.name}
            if variant == "onnx_int8":
                entry[variant]["calibration"] = {
                    "method": args.method_cls if cls else args.method_cd,
                    "samples": args.calib_cls if cls else args.calib_cd}
            print(f"   {variant}: {res}", flush=True)
        models[name] = entry
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    (REPORTS_DIR / "export.md").write_text(render(report))
    print(render(report))


def render(report: dict) -> str:
    lines = ["# Exported models re-scored on the test sets\n",
             f"PyTorch reference on {report['reference_device']} (torch {report['torch']}); "
             f"ONNX on onnxruntime {report['onnxruntime']} CPU; CoreML with compute units ALL. "
             "Latency here is throughput while scoring (ONNX in batches, CoreML batch 1); "
             "see benchmark.md for batch-1 latency.\n"]
    for name, entry in report.get("models", {}).items():
        cls = "accuracy" in entry["pytorch"]
        lines.append(f"\n## {name} ({entry['test_samples']} test "
                     f"{'patches' if cls else 'tiles'})\n")
        if cls:
            lines += ["| Format | MB | Accuracy | Macro F1 | Errors | Same label as PyTorch "
                      "| Max prob diff |", "|---|---|---|---|---|---|---|"]
        else:
            lines += ["| Format | MB | F1 | IoU | Precision | Recall | Pixels flipped vs PyTorch "
                      "| Max prob diff |", "|---|---|---|---|---|---|---|---|"]
        for variant, r in entry.items():
            if not isinstance(r, dict):
                continue
            diff = f"{r['max_prob_diff']:.2g}" if "max_prob_diff" in r else "–"
            if cls:
                agree = f"{100 * r['label_agreement']:.2f} %" if "label_agreement" in r else "–"
                lines.append(f"| {variant} | {r['size_mb']:.1f} | {r['accuracy']:.4f} | "
                             f"{r['macro_f1']:.4f} | {r['errors']} | {agree} | {diff} |")
            else:
                flip = (f"{100 * r['flipped_pixel_share']:.3f} %"
                        if "flipped_pixel_share" in r else "–")
                lines.append(f"| {variant} | {r['size_mb']:.1f} | {r['f1']:.4f} | "
                             f"{r['iou']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                             f"{flip} | {diff} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
