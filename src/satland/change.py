"""Building change detection: Siamese U-Net, a classical CVA baseline, loss and metrics.

Siamese ("absdiff") model: one shared encoder sees the 'before' and 'after' image separately; at
every scale the decoder receives |f_A - f_B|, so features that are identical in both dates cancel
and the network is order-invariant by construction. The "concat" variant stacks both images
into a 6-channel input (early fusion) for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


class SiameseUnet(nn.Module):
    def __init__(self, encoder: str = "resnet18", pretrained: bool = True,
                 fusion: str = "absdiff"):
        super().__init__()
        if fusion not in ("absdiff", "concat"):
            raise ValueError(f"Unknown fusion {fusion!r}")
        self.fusion = fusion
        in_ch = 6 if fusion == "concat" else 3
        base = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                        in_channels=in_ch, classes=1)
        self.encoder, self.decoder, self.head = base.encoder, base.decoder, base.segmentation_head

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            feats = self.encoder(torch.cat([a, b], 1))
        else:
            fa, fb = self.encoder(a), self.encoder(b)
            feats = [torch.abs(x - y) for x, y in zip(fa, fb, strict=True)]
        return self.head(self.decoder(feats))


class BCEDiceLoss(nn.Module):
    """bce_weight * BCE(pos_weight) + (1 - bce_weight) * soft Dice on the change class."""

    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + self.eps) / (p.sum() + target.sum() + self.eps)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


@dataclass
class Confusion:
    """Pixel counts for the 'change' class; pooled over a whole split as in LEVIR-CD papers."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> Confusion:
        pred, target = pred.astype(bool), target.astype(bool)
        self.tp += int((pred & target).sum())
        self.fp += int((pred & ~target).sum())
        self.fn += int((~pred & target).sum())
        self.tn += int((~pred & ~target).sum())
        return self

    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    def f1(self) -> float:
        return 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)

    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)

    def oa(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / max(total, 1)

    def as_dict(self) -> dict[str, float]:
        return {"f1": self.f1(), "iou": self.iou(), "precision": self.precision(),
                "recall": self.recall(), "overall_accuracy": self.oa()}


# ------------------------------------------------------------------- classical baseline
def change_magnitude(a: np.ndarray, b: np.ndarray, blur: int = 2) -> np.ndarray:
    """Change Vector Analysis: per-pixel Euclidean distance between the two RGB vectors.

    Each image is first standardised per channel (crude radiometric normalisation, since the
    dates differ in sun angle and season), then box-blurred to suppress misregistration noise.
    """
    def norm(x):
        x = x.astype(np.float32)
        return (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-6)

    d = np.sqrt(((norm(a) - norm(b)) ** 2).sum(-1))
    if blur > 0:
        k = 2 * blur + 1
        pad = np.pad(d, blur, mode="reflect")
        c = np.cumsum(np.cumsum(pad, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        d = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    return d


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold that maximises between-class variance of a 1-D sample.

    Class 0 = bins 0..k, so the threshold is the *upper edge* of bin k (values > t are class 1).
    """
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[int(np.argmax(between)) + 1])


def save_cd_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_cd_checkpoint(path: Path, device: torch.device | str = "cpu"
                       ) -> tuple[SiameseUnet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SiameseUnet(cfg["encoder"], pretrained=False, fusion=cfg["fusion"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
