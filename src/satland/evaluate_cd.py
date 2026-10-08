"""Evaluate building change detection on the LEVIR-CD test split (2,048 tiles = 128 scenes).

Compares the Siamese U-Net with a classical Change Vector Analysis baseline (unsupervised Otsu
threshold, and a threshold tuned on the validation split). Metrics are pooled over all test
pixels for the 'change' class, as in the LEVIR-CD literature.

Writes reports/cd/results_test.json, reports/cd_gallery.jpg (tiles), reports/cd_scenes.jpg
(full 1024 px scenes stitched from 16 tiles), reports/cd_tile_f1.png and training curves.

Example:
    satland-cd-evaluate --checkpoint models/levircd_absdiff.pt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .change import Confusion, change_magnitude, load_cd_checkpoint, otsu_threshold  # noqa: E402
from .config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from .levircd import ChangeDataset, decode_row, load_names, load_rows  # noqa: E402
from .models import pick_device  # noqa: E402
from .plots import plot_training_curves  # noqa: E402
from .viz import change_overlay, label, mosaic  # noqa: E402

MIN_CHANGE_PX = 50  # a tile "contains change" if it has at least this many changed pixels


@torch.no_grad()
def predict_tiles(model, rows, device, batch_size: int = 32) -> tuple[np.ndarray, float]:
    """Change probabilities (N, H, W) as float16, and mean ms per tile."""
    loader = DataLoader(ChangeDataset(rows), batch_size, num_workers=2)
    out, t_model = [], 0.0
    for a, b, _ in loader:
        a, b = a.to(device), b.to(device)
        t0 = time.perf_counter()
        prob = torch.sigmoid(model(a, b))[:, 0]
        if device.type == "mps":
            torch.mps.synchronize()
        t_model += time.perf_counter() - t0
        out.append(prob.half().cpu().numpy())
    probs = np.concatenate(out)
    return probs, 1000 * t_model / len(probs)


def cva_maps(rows) -> list[np.ndarray]:
    return [change_magnitude(*decode_row(r)[:2]) for r in rows]


def tile_scores(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-tile F1; NaN when the tile has no change and none is predicted (undefined)."""
    tp = (pred & target).sum()
    denom = pred.sum() + target.sum()
    return float(2 * tp / denom) if denom else float("nan")


def stitch(tiles: dict[int, np.ndarray], grid: int = 4) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    out = np.zeros((grid * h, grid * w, *tiles[0].shape[2:]), tiles[0].dtype)
    for k, t in tiles.items():
        out[(k // grid) * h:(k // grid + 1) * h, (k % grid) * w:(k % grid + 1) * w] = t
    return out


def scene_panels(rows, names, preds, scene_ids, size: int = 360) -> list[list[np.ndarray]]:
    """Before / after / ground truth / prediction overlay for whole 1024 px scenes."""
    by_scene = defaultdict(dict)
    for i, n in enumerate(names):
        _, scene, k = re.match(r"(\w+?)_(\d+)_(\d+)$", n).groups()
        by_scene[scene][int(k)] = i
    panels = []
    for sid in scene_ids:
        idx = by_scene[sid]
        decoded = {k: decode_row(rows[i]) for k, i in idx.items()}
        a = stitch({k: d[0] for k, d in decoded.items()})
        b = stitch({k: d[1] for k, d in decoded.items()})
        m = stitch({k: d[2] for k, d in decoded.items()})
        p = stitch({k: preds[i] for k, i in idx.items()})
        conf = Confusion().update(p, m)
        gt = np.repeat((m * 255)[..., None], 3, axis=2).astype(np.uint8)
        tiles = [a, b, gt, change_overlay(b, p, m)]
        tiles = [np.asarray(Image.fromarray(t).resize((size, size), Image.BILINEAR))
                 for t in tiles]
        tiles[0] = label(tiles[0], f"scene {sid}: before")
        tiles[1] = label(tiles[1], "after")
        tiles[2] = label(tiles[2], "ground truth")
        tiles[3] = label(tiles[3], f"Siamese U-Net  F1 {conf.f1():.2f}")
        panels.append(tiles)
    return panels


def tile_gallery(rows, preds, cva_preds, order, size: int = 200) -> np.ndarray:
    out = []
    for i in order:
        a, b, m = decode_row(rows[i])
        gt = np.repeat((m * 255)[..., None], 3, axis=2).astype(np.uint8)
        t = [a, b, gt, change_overlay(b, cva_preds[i], m), change_overlay(b, preds[i], m)]
        t = [np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR)) for x in t]
        cap = ["before", "after", "ground truth",
               f"CVA  F1 {np.nan_to_num(tile_scores(cva_preds[i], m.astype(bool))):.2f}",
               f"Siamese  F1 {np.nan_to_num(tile_scores(preds[i], m.astype(bool))):.2f}"]
        out.append([label(x, c) for x, c in zip(t, cap, strict=True)])
    return mosaic(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "levircd_absdiff.pt")
    ap.add_argument("--extra", nargs="*", type=Path, default=[],
                    help="more checkpoints to score (e.g. the concat model)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    rows, names = load_rows("test"), load_names("test")
    targets = np.stack([decode_row(r)[2] for r in rows]).astype(bool)
    results = {"test_tiles": len(rows), "changed_pixel_share": float(targets.mean()),
               "models": {}}

    preds = None
    for ckpt in [args.checkpoint, *args.extra]:
        model, meta = load_cd_checkpoint(ckpt, device)
        probs, ms = predict_tiles(model, rows, device)
        p = probs > args.threshold
        conf = Confusion().update(p, targets)
        has = targets.sum((1, 2)) >= MIN_CHANGE_PX
        empty_fa = float(((p.sum((1, 2)) >= MIN_CHANGE_PX) & ~has).sum() / max((~has).sum(), 1))
        results["models"][meta["config"]["fusion"]] = {
            **conf.as_dict(), "ms_per_tile": round(ms, 2), "best_epoch": meta["epoch"],
            "val_f1": meta["val_f1"], "empty_tile_false_alarm_rate": empty_fa,
            "params_M": round(sum(x.numel() for x in model.parameters()) / 1e6, 2)}
        print(ckpt.name, results["models"][meta["config"]["fusion"]])
        if preds is None:
            preds = p

    # Classical baseline: CVA magnitude with (a) per-tile Otsu, (b) one threshold tuned on val.
    t0 = time.perf_counter()
    mags = cva_maps(rows)
    cva_ms = 1000 * (time.perf_counter() - t0) / len(rows)
    otsu = np.stack([m > otsu_threshold(m.ravel()) for m in mags])
    val_rows = load_rows("val")
    val_t = np.stack([decode_row(r)[2] for r in val_rows]).astype(bool)
    val_m = np.stack(cva_maps(val_rows))
    grid = np.linspace(0.5, 4.0, 36)
    f1s = [Confusion().update(val_m > g, val_t).f1() for g in grid]
    best_t = float(grid[int(np.argmax(f1s))])
    tuned = np.stack(mags) > best_t
    results["models"]["cva_otsu"] = {**Confusion().update(otsu, targets).as_dict(),
                                     "ms_per_tile": round(cva_ms, 2)}
    results["models"]["cva_val_tuned"] = {**Confusion().update(tuned, targets).as_dict(),
                                          "threshold": best_t, "ms_per_tile": round(cva_ms, 2)}
    for k in ("cva_otsu", "cva_val_tuned"):
        print(k, results["models"][k])

    # Per-tile F1 distribution for tiles that contain change.
    has = targets.sum((1, 2)) >= MIN_CHANGE_PX
    deep_f1 = np.array([tile_scores(preds[i], targets[i]) for i in np.flatnonzero(has)])
    cva_f1 = np.array([tile_scores(tuned[i], targets[i]) for i in np.flatnonzero(has)])
    results["tiles_with_change"] = int(has.sum())
    results["median_tile_f1"] = {"siamese": float(np.nanmedian(deep_f1)),
                                 "cva_val_tuned": float(np.nanmedian(cva_f1))}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist([cva_f1, deep_f1], bins=20, range=(0, 1), label=["CVA (val-tuned)", "Siamese U-Net"],
            color=["#9aa5b1", "#3d7ea6"])
    ax.set_xlabel(f"per-tile F1 (tiles with ≥ {MIN_CHANGE_PX} changed pixels, n = {has.sum()})")
    ax.set_ylabel("tiles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "cd_tile_f1.png", dpi=130)
    plt.close(fig)

    # Galleries: best / median / worst change tiles, then 3 whole scenes.
    idx = np.flatnonzero(has)
    order = idx[np.argsort(np.nan_to_num(deep_f1))]
    pick = [order[-1], order[-len(order) // 4], order[len(order) // 2], order[len(order) // 6],
            order[0]]
    Image.fromarray(tile_gallery(rows, preds, tuned, pick)).save(
        REPORTS_DIR / "cd_gallery.jpg", quality=90)
    scene_change = defaultdict(int)
    for i, n in enumerate(names):
        scene_change[n.split("_")[1]] += int(targets[i].sum())
    busiest = sorted(scene_change, key=scene_change.get, reverse=True)
    scenes = [busiest[0], busiest[len(busiest) // 5], busiest[len(busiest) // 3]]
    Image.fromarray(mosaic(scene_panels(rows, names, preds, scenes))).save(
        REPORTS_DIR / "cd_scenes.jpg", quality=90)

    hists = {}
    for p in sorted((REPORTS_DIR / "cd").glob("history_*.json")):
        hists[p.stem.replace("history_", "")] = json.loads(p.read_text())
    if hists:
        plot_training_curves(hists, REPORTS_DIR / "training_curves_cd.png", "val_f1",
                             "LEVIR-CD change detection")
    (REPORTS_DIR / "cd").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cd" / "results_test.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "models"}, indent=1))


if __name__ == "__main__":
    main()
