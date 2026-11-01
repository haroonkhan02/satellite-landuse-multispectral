"""Batch-1 latency per backend: PyTorch (MPS / CPU), ONNX Runtime CPU, CoreML.

Times the network alone on one real test input per model (a EuroSAT patch, a LEVIR-CD tile
pair): warm-up runs first, then the median and 90th percentile.

Example:
    satland-benchmark --runs 100
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .config import REPORTS_DIR
from .eurosat import load_cache, load_split
from .export import CoreMLRunner, OnnxRunner, TorchRunner, load_change, load_classifier
from .export_models import EXPORT_DIR, MODELS, VARIANTS, cd_feed, file_mb, file_name
from .levircd import load_rows


def time_fn(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def sample_feed(kind: str) -> dict[str, np.ndarray]:
    if kind == "cls":
        cache, files, _ = load_cache()
        i = int(load_split(files)["test"][0])
        return {"bands": np.asarray(cache[i:i + 1]).astype(np.float32)}
    return cd_feed(load_rows("test")[:1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["eurosat_all13", "levircd_siamese"],
                    choices=list(MODELS))
    ap.add_argument("--dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args(argv)
    import onnxruntime as ort

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    report = {"machine": f"{platform.machine()} {platform.platform()}",
              "onnxruntime": ort.__version__, "torch": torch.__version__, "runs": args.runs,
              "models": {}}
    for name in args.models:
        kind, ckpt = MODELS[name]
        feed = sample_feed(kind)
        rows = {}
        for dev in devices:
            wrapper = load_classifier(ckpt)[0] if kind == "cls" else load_change(ckpt)
            runner = TorchRunner(wrapper, dev)
            rows[f"pytorch_fp32 ({dev})"] = {
                "size_mb": file_mb(ckpt),
                **time_fn(lambda r=runner, f=feed: r.run(f), args.runs, args.warmup)}
        for variant in VARIANTS:
            path = args.dir / file_name(name, variant)
            if not path.exists():
                continue
            units = ("",) if variant.startswith("onnx") else ("ALL", "CPU_AND_GPU", "CPU_ONLY")
            for u in units:
                runner = OnnxRunner(path) if not u else CoreMLRunner(path, u)
                label = f"{variant} ({'onnxruntime CPU' if not u else u.lower()})"
                rows[label] = {"size_mb": file_mb(path),
                               **time_fn(lambda r=runner, f=feed: r.run(f), args.runs, args.warmup)}
        shape = " + ".join("x".join(map(str, v.shape)) for v in feed.values())
        report["models"][name] = {"input": shape, "results": rows}
        print(name, json.dumps(rows, indent=1), flush=True)

    lines = ["# Model latency per backend\n",
             f"{report['machine']}; batch 1; median and 90th percentile of {args.runs} runs "
             f"after {args.warmup} warm-up runs, network only.\n"]
    for name, m in report["models"].items():
        lines += [f"\n## {name} (input {m['input']})\n",
                  "| Backend | Size (MB) | Median (ms) | p90 (ms) |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['size_mb']:.1f} | {v['median_ms']:.2f} | {v['p90_ms']:.2f} |"
                  for k, v in m["results"].items()]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))
    (REPORTS_DIR / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
