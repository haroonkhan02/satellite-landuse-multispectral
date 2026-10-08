"""End-to-end smoke tests on the synthetic fixtures (CPU, a few seconds)."""

import numpy as np
import torch

from satland.ablation import image_features, write_markdown
from satland.demo import detect_change
from satland.plots import plot_ablation
from tests.conftest import change_pair, synthetic_cube


def test_image_features_shapes():
    raw = np.stack([synthetic_cube(seed=i) for i in range(3)])
    assert image_features(raw, "rgb").shape == (3, 12)
    assert image_features(raw, "all13").shape == (3, 13 * 4 + 8)


def test_ablation_report_and_plot(tmp_path):
    results = [
        {"model": "random forest (band stats)", "bands": "rgb", "train_fraction": 1.0,
         "features": 12, "accuracy": 0.8, "macro_f1": 0.79, "train_minutes": 0.1},
        {"model": "ResNet-18 (ImageNet)", "bands": "rgb", "channels": 3, "train_fraction": 1.0,
         "accuracy": 0.97, "macro_f1": 0.97, "train_minutes": 4},
        {"model": "ResNet-18 (ImageNet)", "bands": "rgb", "channels": 3, "train_fraction": 0.1,
         "accuracy": 0.9, "macro_f1": 0.9, "train_minutes": 2},
        {"model": "ResNet-18 (ImageNet)", "bands": "all13", "channels": 13,
         "train_fraction": 0.1, "accuracy": 0.95, "macro_f1": 0.95, "train_minutes": 2},
    ]
    md = write_markdown(results)
    assert md.count("\n") == 6 and "| 10% |" in md
    assert plot_ablation(results, tmp_path / "a.png").stat().st_size > 1000


def test_full_image_change_detection_pads_odd_sizes():
    from satland.change import SiameseUnet

    model = SiameseUnet("resnet18", pretrained=False).eval()
    a, b, _ = change_pair(50, seed=0)
    pred = detect_change(model, a, b, torch.device("cpu"))
    assert pred.shape == (50, 50) and pred.dtype == bool
