"""Evaluate EuroSAT classifiers on the test split and explain what the extra bands buy.

Writes to reports/:
    cls/results_test.json         accuracy, macro F1, per-class F1 for each model, band importance
    confusion_<bands>.png         confusion matrices
    per_class_f1.png              per-class F1, RGB vs all-13
    spectral_signatures.png       mean reflectance per class across the 13 bands
    band_importance.png           accuracy drop when one band is blanked (all-13 model)
    gallery_test.jpg              RGB / CIR / SWIR / NDVI views of the RGB model's mistakes

Example:
    satland-evaluate --models all13 rgb
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from .bands import BAND_NAMES, REFLECTANCE_SCALE  # noqa: E402
from .config import EUROSAT_CLASSES, MODELS_DIR, REPORTS_DIR  # noqa: E402
from .eurosat import STATS_FILE, InputBuilder, load_cache, load_split  # noqa: E402
from .models import load_checkpoint, pick_device  # noqa: E402
from .plots import (  # noqa: E402
    plot_band_importance,
    plot_confusion,
    plot_spectral_signatures,
    plot_training_curves,
)
from .train_cls import predict, run_name, scores  # noqa: E402
from .viz import label, mosaic, upscale, views  # noqa: E402

SHORT = {"HerbaceousVegetation": "Herbaceous", "PermanentCrop": "PermCrop"}


@torch.no_grad()
def band_occlusion(model, meta, stats, cache, labels, idx, device) -> dict[str, float]:
    """Accuracy drop when each input band is replaced by its training mean (0 after scaling)."""
    builder = InputBuilder(meta["band_idx"], meta["with_indices"], stats, device)
    base_probs, y, _ = predict(model, builder, cache, labels, idx, device)
    base = scores(base_probs, y)["accuracy"]
    drops = {}
    for pos, band_i in enumerate(meta["band_idx"]):
        def blanked(raw, p=pos):
            x = builder(raw)
            x[:, p] = 0.0
            return x

        probs, _, _ = predict(model, blanked, cache, labels, idx, device)
        drops[BAND_NAMES[band_i]] = base - scores(probs, y)["accuracy"]
    return {"base_accuracy": base, "drops": drops}


def class_spectra(cache, labels, idx) -> tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for c in range(len(EUROSAT_CLASSES)):
        sel = idx[labels[idx] == c]
        x = np.asarray(cache[sel]).astype(np.float32) / REFLECTANCE_SCALE
        per_img = x.mean(axis=(2, 3))
        means.append(per_img.mean(0))
        stds.append(per_img.std(0))
    return np.stack(means), np.stack(stds)


def mistakes_gallery(cache, ids, y, pred_rgb, pred_all, n: int = 6, size: int = 160,
                     seed: int = 0) -> np.ndarray:
    """Images the RGB model gets wrong and the all-13 model gets right, in four views."""
    rng = np.random.default_rng(seed)
    fixed = np.flatnonzero((pred_rgb != y) & (pred_all == y))
    chosen, seen = [], set()
    for i in rng.permutation(fixed):  # one example per confused class pair, for variety
        pair = (y[i], pred_rgb[i])
        if pair not in seen:
            seen.add(pair)
            chosen.append(i)
        if len(chosen) == n:
            break
    rows = []
    for i in chosen:
        cube = np.asarray(cache[ids[i]])
        tiles = [upscale(v, size) for v in views(cube)]
        wrong = EUROSAT_CLASSES[pred_rgb[i]]
        rgb_txt = f"RGB model: {SHORT.get(wrong, wrong)}"
        tiles[0] = label(tiles[0], f"true: {EUROSAT_CLASSES[y[i]]}")
        tiles[1] = label(tiles[1], rgb_txt, color=(255, 120, 120))
        tiles[2] = label(tiles[2], "13-band model: correct", color=(120, 255, 140))
        tiles[3] = label(tiles[3], "NDVI")
        rows.append(tiles)
    header = [label(np.full((18, size, 3), 255, np.uint8), t, (0, 0, 0), (255, 255, 255))
              for t in ("true colour (B4,B3,B2)", "colour IR (B8,B4,B3)", "SWIR (B12,B8,B4)",
                        "NDVI")]
    return mosaic([header] + rows) if rows else np.zeros((10, 10, 3), np.uint8)


def per_class_plot(f1s: dict[str, list[float]], path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(EUROSAT_CLASSES))
    width = 0.8 / len(f1s)
    for k, (name, vals) in enumerate(f1s.items()):
        ax.bar(x + k * width - 0.4 + width / 2, vals, width, label=name)
    ax.set_xticks(x, [SHORT.get(c, c) for c in EUROSAT_CLASSES], rotation=30, ha="right")
    lo = min(min(v) for v in f1s.values())
    ax.set_ylim(max(0, lo - 0.03), 1.0)
    ax.set_ylabel("test F1")
    ax.set_title("Per-class F1 by band set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["rgb", "rgb_nir", "all13"],
                    help="band-set names of trained models/eurosat_<bands>.pt")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    cache, files, labels = load_cache()
    split = load_split(files)
    test = split["test"]
    stats = json.loads(STATS_FILE.read_text())
    out = {"test_images": int(len(test)), "models": {}}
    preds, f1s, histories = {}, {}, {}
    for bands in args.models:
        name = run_name(bands)
        model, meta = load_checkpoint(MODELS_DIR / f"eurosat_{name}.pt", device)
        builder = InputBuilder(meta["band_idx"], meta["with_indices"], stats, device)
        probs, y, ids = predict(model, builder, cache, labels, test, device)
        pred = probs.argmax(1)
        preds[bands] = pred
        f1s[bands] = f1_score(y, pred, average=None).tolist()
        cm = confusion_matrix(y, pred, labels=range(len(EUROSAT_CLASSES)))
        plot_confusion(cm, [SHORT.get(c, c) for c in EUROSAT_CLASSES],
                       REPORTS_DIR / f"confusion_{name}.png",
                       f"{bands}: test accuracy {scores(probs, y)['accuracy']:.4f}")
        out["models"][bands] = {**scores(probs, y), "errors": int((pred != y).sum()),
                                "per_class_f1": dict(zip(EUROSAT_CLASSES, f1s[bands],
                                                         strict=True)),
                                "confusion": cm.tolist(), "best_epoch": meta["epoch"]}
        hist_path = REPORTS_DIR / "cls" / f"history_{name}.json"
        if hist_path.exists():
            histories[bands] = json.loads(hist_path.read_text())
        if bands == "all13":
            occ = band_occlusion(model, meta, stats, cache, labels, test, device)
            out["band_occlusion"] = occ
            plot_band_importance(list(occ["drops"]), list(occ["drops"].values()),
                                 occ["base_accuracy"], REPORTS_DIR / "band_importance.png")
        print(f"{bands:>12}: acc {out['models'][bands]['accuracy']:.4f}  "
              f"macro F1 {out['models'][bands]['macro_f1']:.4f}  errors {(pred != y).sum()}")
    per_class_plot(f1s, REPORTS_DIR / "per_class_f1.png")
    means, stds = class_spectra(cache, labels, test)
    plot_spectral_signatures(means, stds, list(EUROSAT_CLASSES),
                             REPORTS_DIR / "spectral_signatures.png")
    if histories:
        plot_training_curves(histories, REPORTS_DIR / "training_curves_cls.png", "val_acc",
                             "EuroSAT ResNet-18 by band set")
    if "rgb" in preds and "all13" in preds:
        y = labels[np.sort(test)]
        out["rgb_wrong_all13_right"] = int(((preds["rgb"] != y) & (preds["all13"] == y)).sum())
        out["all13_wrong_rgb_right"] = int(((preds["all13"] != y) & (preds["rgb"] == y)).sum())
        gal = mistakes_gallery(cache, np.sort(test), y, preds["rgb"], preds["all13"])
        Image.fromarray(gal).save(REPORTS_DIR / "gallery_test.jpg", quality=92)
    (REPORTS_DIR / "cls").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cls" / "results_test.json").write_text(json.dumps(out, indent=1))
    print(f"wrote reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
