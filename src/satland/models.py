"""Classifier with an N-channel stem, pretrained-weight transfer and checkpoints.

ImageNet weights only exist for 3 RGB channels. For a multispectral input the first conv is
rebuilt with one filter slice per input channel: Sentinel-2 red/green/blue (B04/B03/B02) get the
matching pretrained slice, every other band (and every spectral index) gets the mean RGB slice,
and the whole kernel is rescaled by 3 / C so the first activations keep their pretrained scale.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .bands import RGB_BANDS


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def adapt_stem_weight(rgb_weight: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
    """(out, 3, k, k) ImageNet kernel -> (out, C, k, k) kernel for the named input channels."""
    mean_slice = rgb_weight.mean(dim=1)
    slices = []
    for name in channel_names:
        if name in RGB_BANDS:
            slices.append(rgb_weight[:, RGB_BANDS.index(name)])
        else:
            slices.append(mean_slice)
    weight = torch.stack(slices, dim=1)
    return weight * (3.0 / len(channel_names))


def build_classifier(channel_names: list[str], num_classes: int, arch: str = "resnet18",
                     pretrained: bool = True) -> nn.Module:
    """torchvision ResNet with a C-channel stem and a ``num_classes`` head."""
    if arch not in ("resnet18", "resnet34", "resnet50"):
        raise ValueError(f"Unsupported arch {arch!r}")
    weights = "DEFAULT" if pretrained else None
    net = getattr(models, arch)(weights=weights)
    old = net.conv1
    net.conv1 = nn.Conv2d(len(channel_names), old.out_channels, kernel_size=old.kernel_size,
                          stride=old.stride, padding=old.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            net.conv1.weight.copy_(adapt_stem_weight(old.weight, channel_names))
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def save_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the classifier from the metadata stored next to its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(ckpt["channel_names"], len(ckpt["classes"]), ckpt["config"]["arch"],
                             pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
