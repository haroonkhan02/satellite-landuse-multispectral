"""Matplotlib figures for the reports (headless backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bands import BANDS  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ablation(results: list[dict], path: Path) -> Path:
    """Left: test accuracy per band set (full labels). Right: RGB vs all-13 as labels shrink."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    full = [r for r in results if r["train_fraction"] == 1.0]
    labels = [f"{'RF' if 'forest' in r['model'] else 'CNN'} · {r['bands']}" for r in full]
    accs = [r["accuracy"] for r in full]
    colors = ["#9aa5b1" if "forest" in r["model"] else "#3d7ea6" for r in full]
    y = np.arange(len(full))
    ax1.barh(y, accs, color=colors)
    for yi, a in zip(y, accs, strict=True):
        ax1.text(a + 0.002, yi, f"{a:.3f}", va="center", fontsize=8)
    ax1.set_yticks(y, labels, fontsize=8)
    ax1.invert_yaxis()
    lo = min(accs) if accs else 0.8
    ax1.set_xlim(max(0.0, lo - 0.05), 1.0)
    ax1.set_xlabel("test accuracy (5,400 images)")
    ax1.set_title("Band sets, 100 % of training labels")

    for bands, color in (("rgb", "#d1495b"), ("all13", "#2e8b57")):
        pts = sorted((r["train_fraction"], r["accuracy"]) for r in results
                     if r["bands"] == bands and "forest" not in r["model"])
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax2.plot(xs, ys, "o-", color=color, label=f"CNN · {bands}")
            for x, v in pts:
                ax2.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(4, -10),
                             fontsize=8, color=color)
    ax2.set_xscale("log")
    ax2.set_xticks([0.01, 0.1, 1.0], ["1 %", "10 %", "100 %"])
    ax2.set_xlabel("share of the 18,900 training labels")
    ax2.set_ylabel("test accuracy")
    ax2.set_title("Do extra bands help more with few labels?")
    ax2.grid(alpha=0.3)
    ax2.legend()
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> Path:
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return _save(fig, path)


def plot_spectral_signatures(means: np.ndarray, stds: np.ndarray, classes: list[str],
                             path: Path) -> Path:
    """Mean reflectance per class across the 13 bands, ordered by wavelength."""
    order = np.argsort([b.wavelength_nm for b in BANDS])
    wl = np.array([BANDS[i].wavelength_nm for i in order])
    names = [BANDS[i].name for i in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for c, cls in enumerate(classes):
        m, s = means[c, order], stds[c, order]
        ax.plot(wl, m, "o-", ms=3, color=cmap(c), label=cls)
        ax.fill_between(wl, m - s, m + s, color=cmap(c), alpha=0.07)
    for x, n in zip(wl, names, strict=True):
        ax.annotate(n, (x, 0), xytext=(0, 2), textcoords="offset points", fontsize=7,
                    ha="center", color="grey")
    ax.axvspan(400, 700, color="gold", alpha=0.06)
    ax.text(550, ax.get_ylim()[1] * 0.95, "visible", ha="center", fontsize=8, color="goldenrod")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("mean TOA reflectance (test split)")
    ax.set_title("Spectral signatures of the EuroSAT classes")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_band_importance(names: list[str], drops: list[float], base_acc: float,
                         path: Path) -> Path:
    order = np.argsort(drops)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([names[i] for i in order], [drops[i] for i in order], color="#3d7ea6")
    ax.set_ylabel("accuracy drop when band is blanked")
    ax.set_title(f"Band occlusion importance, all-13 model (base accuracy {base_acc:.3f})")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, path)


def plot_training_curves(histories: dict[str, dict], path: Path, metric: str,
                         title: str) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h["epochs"]]
        ax1.plot(ep, [r["train_loss"] for r in h["epochs"]], label=name)
        ax2.plot(ep, [r[metric] for r in h["epochs"]], label=name)
    ax1.set_title("training loss")
    ax2.set_title(metric.replace("_", " "))
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)
