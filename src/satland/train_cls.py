"""Train a EuroSAT land-use classifier on a chosen Sentinel-2 band set.

The checkpoint with the best validation accuracy is kept as models/eurosat_<bands>.pt and the
per-epoch history goes to reports/cls/history_<bands>.json.

Examples:
    satland-train --bands all13
    satland-train --bands rgb --epochs 12
    satland-train --bands rgb_nir+idx       # + NDVI / NDWI / NDBI / NDMI channels
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .bands import INDEX_NAMES, band_indices, resolve_band_set
from .config import CONFIGS_DIR, EUROSAT_CLASSES, MODELS_DIR, REPORTS_DIR, ClsConfig
from .eurosat import (
    STATS_FILE,
    InputBuilder,
    augment_batch,
    iterate_batches,
    load_cache,
    load_split,
)
from .models import build_classifier, pick_device, save_checkpoint

DEFAULT_CONFIG = CONFIGS_DIR / "eurosat_resnet18.yaml"


def run_name(bands: str, train_fraction: float = 1.0) -> str:
    name = bands.replace("+", "_plus_").replace(",", "-")
    return name if train_fraction >= 1.0 else f"{name}_frac{train_fraction:g}"


def subsample(idx: np.ndarray, labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Stratified ``fraction`` of ``idx`` (at least one image per class)."""
    if fraction >= 1.0:
        return idx
    rng = np.random.default_rng(seed + 1)
    keep = []
    for c in np.unique(labels[idx]):
        members = idx[labels[idx] == c]
        n = max(1, round(fraction * len(members)))
        keep.append(rng.choice(members, n, replace=False))
    return np.sort(np.concatenate(keep))


def channel_names(bands: str) -> tuple[list[str], list[int], bool]:
    names, with_idx = resolve_band_set(bands)
    return list(names) + (list(INDEX_NAMES) if with_idx else []), band_indices(names), with_idx


@torch.no_grad()
def predict(model: nn.Module, builder: Callable[[torch.Tensor], torch.Tensor], cache, labels,
            idx, device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax probabilities, labels and image indices for ``idx`` (in sorted order).

    ``builder`` maps a raw uint16 batch to model input (an InputBuilder, or a wrapper of one).
    """
    model.eval()
    probs, ys, ids = [], [], []
    for raw, y, b in iterate_batches(cache, labels, idx, batch_size):
        logits = model(builder(raw.to(device)))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
        ids.append(b)
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(ids)


def scores(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    return {"accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


def train(cfg: ClsConfig, out_dir: Path = MODELS_DIR, report_dir: Path = REPORTS_DIR / "cls",
          device: str | None = None, max_batches: int | None = None) -> dict:
    """Train one band set; returns the history dict (also written to ``report_dir``)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = pick_device(device)
    cache, files, labels = load_cache()
    split = load_split(files)
    train_idx = subsample(split["train"], labels, cfg.train_fraction, cfg.seed)
    stats = json.loads(STATS_FILE.read_text())
    names, idx, with_idx = channel_names(cfg.bands)
    builder = InputBuilder(idx, with_idx, stats, dev)
    model = build_classifier(names, len(EUROSAT_CLASSES), cfg.arch, cfg.pretrained).to(dev)

    steps = int(np.ceil(len(train_idx) / cfg.batch_size))
    if max_batches:
        steps = min(steps, max_batches)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total,
                                                pct_start=min(0.5, max(0.15, 2 / total)))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    name = run_name(cfg.bands, cfg.train_fraction)
    ckpt_path = out_dir / f"eurosat_{name}.pt"
    history = {"config": cfg.to_dict(), "channels": names, "device": str(dev),
               "train_images": len(train_idx), "epochs": []}
    best = -1.0
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        batches = iterate_batches(cache, labels, train_idx, cfg.batch_size, True, rng)
        for step, (raw, y, _) in enumerate(batches):
            if step >= steps:
                break
            x = augment_batch(builder(raw.to(dev)))
            loss = criterion(model(x), y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        probs, y_val, _ = predict(model, builder, cache, labels, split["val"], dev)
        val = scores(probs, y_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "seconds": round(time.time() - t0, 1)}
        history["epochs"].append(row)
        report_dir.mkdir(parents=True, exist_ok=True)  # written every epoch: survives a kill
        (report_dir / f"history_{name}.json").write_text(json.dumps(history, indent=1))
        print(f"[{cfg.bands}] epoch {epoch:2d}/{cfg.epochs}  loss {row['train_loss']:.4f}  "
              f"val acc {val['accuracy']:.4f}  f1 {val['macro_f1']:.4f}  ({row['seconds']} s)",
              flush=True)
        if val["accuracy"] > best:
            best = val["accuracy"]
            save_checkpoint(model, ckpt_path, {
                "config": cfg.to_dict(), "channel_names": names, "band_idx": idx,
                "with_indices": with_idx, "stats": stats, "classes": list(EUROSAT_CLASSES),
                "epoch": epoch, "val_acc": best})
    history["best_val_acc"] = best
    history["train_minutes"] = round((time.time() - t_start) / 60, 2)
    history["checkpoint"] = ckpt_path.name
    (report_dir / f"history_{name}.json").write_text(json.dumps(history, indent=1))
    return history


def build_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--arch")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", help="mps / cuda / cpu (default: best available)")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--bands", help="band set, e.g. rgb, rgb_nir, all13, all13+idx, B04,B08")
    ap.add_argument("--train-fraction", type=float, help="stratified share of the train split")
    args = ap.parse_args(argv)
    cfg = ClsConfig.load(args.config if args.config.exists() else None).override(
        bands=args.bands, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        arch=args.arch, pretrained=False if args.no_pretrained else None,
        train_fraction=args.train_fraction)
    hist = train(cfg, device=args.device)
    print(f"best val acc {hist['best_val_acc']:.4f} in {hist['train_minutes']} min "
          f"-> models/{hist['checkpoint']}")


if __name__ == "__main__":
    main()
