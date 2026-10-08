"""Band-ablation study: which Sentinel-2 bands matter, and when?

Trains one ResNet-18 per band set (full training split), repeats RGB vs all-13 with 1 % and 10 %
of the labels, adds a random-forest baseline on per-image band statistics (spectra only, no
spatial context), and scores every model on the held-out test split.

Outputs reports/ablation.json, reports/ablation.md and reports/band_ablation.png. Finished runs
are reused (their checkpoint + history exist) unless --retrain is given.

Example:
    satland-ablation                     # ~30 min on an Apple M5
    satland-ablation --quick             # 1 epoch per run, smoke test
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .bands import band_indices, resolve_band_set, spectral_indices
from .config import MODELS_DIR, REPORTS_DIR, ClsConfig
from .eurosat import STATS_FILE, InputBuilder, load_cache, load_split
from .models import load_checkpoint, pick_device
from .train_cls import DEFAULT_CONFIG, predict, run_name, scores, subsample, train

# (band set, train fraction, epochs, batch size). Small fractions get more epochs and smaller
# batches so every run sees a comparable number of optimiser steps.
EXPERIMENTS: list[tuple[str, float, int, int]] = [
    ("rgb", 1.0, 8, 128),
    ("rgb_nir", 1.0, 8, 128),
    ("rgb_nir_swir", 1.0, 8, 128),
    ("no_rgb", 1.0, 8, 128),
    ("all13", 1.0, 8, 128),
    ("all13+idx", 1.0, 8, 128),
    ("rgb", 0.1, 30, 64),
    ("all13", 0.1, 30, 64),
    ("rgb", 0.01, 60, 32),
    ("all13", 0.01, 60, 32),
]
RF_BAND_SETS = ("rgb", "rgb_nir", "all13")


def image_features(raw: np.ndarray, bands: str) -> np.ndarray:
    """Per-image mean, std and 10/90th percentiles of each band (+ index means if NIR & SWIR)."""
    names, _ = resolve_band_set(bands)
    x = raw.astype(np.float32) / 10_000.0
    sel = x[:, band_indices(names)]
    flat = sel.reshape(sel.shape[0], sel.shape[1], -1)
    feats = [flat.mean(-1), flat.std(-1), np.percentile(flat, 10, -1), np.percentile(flat, 90, -1)]
    if {"B08", "B11"} <= set(names):
        idx = spectral_indices(x).reshape(x.shape[0], 4, -1)
        feats += [idx.mean(-1), idx.std(-1)]
    return np.concatenate(feats, axis=1)


def run_random_forest(cache, labels, split, bands: str, seed: int = 0) -> dict:
    t0 = time.time()
    tr = image_features(np.asarray(cache[split["train"]]), bands)
    te = image_features(np.asarray(cache[split["test"]]), bands)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(tr, labels[split["train"]])
    probs = rf.predict_proba(te)
    return {"model": "random forest (band stats)", "bands": bands, "train_fraction": 1.0,
            "features": int(tr.shape[1]), **scores(probs, labels[split["test"]]),
            "train_minutes": round((time.time() - t0) / 60, 2)}


def run_cnn(bands: str, fraction: float, epochs: int, batch: int, base: ClsConfig, device,
            cache, labels, split, retrain: bool, quick: bool) -> dict:
    cfg = base.override(bands=bands, train_fraction=fraction, epochs=1 if quick else epochs,
                        batch_size=batch)
    name = run_name(bands, fraction)
    ckpt = MODELS_DIR / f"eurosat_{name}.pt"
    hist_path = REPORTS_DIR / "cls" / f"history_{name}.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    if retrain or quick or not ckpt.exists() or "train_minutes" not in hist:  # unfinished run
        hist = train(cfg, device=device)
    else:
        print(f"reusing {ckpt.name}")
    model, meta = load_checkpoint(ckpt, device)
    stats = json.loads(STATS_FILE.read_text())
    builder = InputBuilder(meta["band_idx"], meta["with_indices"], stats, device)
    probs, y, _ = predict(model, builder, cache, labels, split["test"], device)
    return {"model": f"ResNet-18 ({'ImageNet' if cfg.pretrained else 'scratch'})",
            "bands": bands, "channels": len(meta["channel_names"]), "train_fraction": fraction,
            "train_images": hist.get("train_images", len(subsample(split["train"], labels,
                                                                   fraction, cfg.seed))),
            "epochs": cfg.epochs, "best_epoch": meta["epoch"], "val_acc": meta["val_acc"],
            **scores(probs, y), "train_minutes": hist["train_minutes"]}


def write_markdown(results: list[dict]) -> str:
    lines = ["| Model | Bands | Channels | Labels used | Test acc | Macro F1 | Train min |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        frac = f"{r['train_fraction']:.0%}"
        lines.append(f"| {r['model']} | `{r['bands']}` | {r.get('channels', r.get('features'))} "
                     f"| {frac} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
                     f"| {r['train_minutes']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 epoch per run (smoke test)")
    ap.add_argument("--only", nargs="*", help="subset of band sets to run")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    base = ClsConfig.load(DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    cache, files, labels = load_cache()
    split = load_split(files)
    results = []
    for bands in RF_BAND_SETS:
        if args.only and bands not in args.only:
            continue
        results.append(run_random_forest(cache, labels, split, bands))
        print(results[-1])
    for bands, frac, epochs, batch in EXPERIMENTS:
        if args.only and bands not in args.only:
            continue
        results.append(run_cnn(bands, frac, epochs, batch, base, device, cache, labels, split,
                               args.retrain, args.quick))
        print({k: results[-1][k] for k in ("bands", "train_fraction", "accuracy", "macro_f1")})
    out = REPORTS_DIR / ("ablation_quick" if args.quick else "ablation")
    out.with_suffix(".json").write_text(json.dumps({"device": str(device), "results": results},
                                                   indent=1))
    out.with_suffix(".md").write_text(write_markdown(results))
    from .plots import plot_ablation

    plot_ablation(results, REPORTS_DIR / "band_ablation.png")
    print(write_markdown(results))


if __name__ == "__main__":
    main()
