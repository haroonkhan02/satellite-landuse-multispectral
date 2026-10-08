"""Train the Siamese U-Net for building change detection on LEVIR-CD (256 px tiles).

The checkpoint with the best validation F1 (change class, pooled pixels) is kept as
models/levircd_<fusion>.pt; the history goes to reports/cd/history_<fusion>.json.

Examples:
    satland-cd-train                         # Siamese |fA - fB|, ResNet-18 encoder
    satland-cd-train --fusion concat         # early-fusion baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .change import BCEDiceLoss, Confusion, SiameseUnet, save_cd_checkpoint
from .config import CONFIGS_DIR, MODELS_DIR, REPORTS_DIR, CDConfig
from .levircd import ChangeDataset, load_rows
from .models import pick_device

DEFAULT_CONFIG = CONFIGS_DIR / "levircd_siamese.yaml"


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    conf = Confusion()
    for a, b, m in loader:
        prob = torch.sigmoid(model(a.to(device), b.to(device)))
        conf.update((prob > threshold).cpu().numpy(), m.numpy() > 0.5)
    return conf.as_dict()


def train(cfg: CDConfig, device: str | None = None, out_dir: Path = MODELS_DIR,
          report_dir: Path = REPORTS_DIR / "cd") -> dict:
    torch.manual_seed(cfg.seed)
    dev = pick_device(device)
    train_rows = load_rows("train")
    if cfg.max_train:
        train_rows = train_rows[:cfg.max_train]
    train_ds = ChangeDataset(train_rows, augment=True, seed=cfg.seed,
                             swap_prob=0.5 if cfg.fusion == "absdiff" else 0.0)
    val_ds = ChangeDataset(load_rows("val"))
    kw = {"num_workers": cfg.workers, "persistent_workers": cfg.workers > 0}
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, 32, **kw)

    model = SiameseUnet(cfg.encoder, cfg.pretrained, cfg.fusion).to(dev)
    criterion = BCEDiceLoss(cfg.bce_weight, cfg.pos_weight).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total,
                                                pct_start=min(0.5, max(0.1, 2 / total)))
    ckpt_path = out_dir / f"levircd_{cfg.fusion}.pt"
    history = {"config": cfg.to_dict(), "device": str(dev), "train_tiles": len(train_ds),
               "epochs": []}
    best, t_start = -1.0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)
        t0, losses = time.time(), []
        for a, b, m in train_dl:
            loss = criterion(model(a.to(dev), b.to(dev)), m.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = evaluate_loader(model, val_dl, dev)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               **{f"val_{k}": v for k, v in val.items()}, "seconds": round(time.time() - t0, 1)}
        history["epochs"].append(row)
        report_dir.mkdir(parents=True, exist_ok=True)  # written every epoch: survives a kill
        (report_dir / f"history_{cfg.fusion}.json").write_text(json.dumps(history, indent=1))
        print(f"[cd/{cfg.fusion}] epoch {epoch:2d}/{cfg.epochs}  loss {row['train_loss']:.4f}  "
              f"val F1 {val['f1']:.4f}  IoU {val['iou']:.4f}  ({row['seconds']} s)",
              flush=True)
        if val["f1"] > best:
            best = val["f1"]
            save_cd_checkpoint(model, ckpt_path, {"config": cfg.to_dict(), "epoch": epoch,
                                                  "val_f1": best})
    history["best_val_f1"] = best
    history["train_minutes"] = round((time.time() - t_start) / 60, 2)
    history["checkpoint"] = ckpt_path.name
    (report_dir / f"history_{cfg.fusion}.json").write_text(json.dumps(history, indent=1))
    return history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--fusion", choices=["absdiff", "concat"])
    ap.add_argument("--encoder")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-train", type=int, help="use only the first N training tiles")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = CDConfig.load(args.config if args.config.exists() else None).override(
        fusion=args.fusion, encoder=args.encoder, epochs=args.epochs,
        batch_size=args.batch_size, max_train=args.max_train)
    hist = train(cfg, args.device)
    print(f"best val F1 {hist['best_val_f1']:.4f} in {hist['train_minutes']} min "
          f"-> models/{hist['checkpoint']}")


if __name__ == "__main__":
    main()
