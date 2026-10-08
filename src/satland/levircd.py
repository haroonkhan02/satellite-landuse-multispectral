"""LEVIR-CD bi-temporal tiles: parquet loading, paired augmentation and a torch Dataset.

Each row holds encoded bytes for image A (before), image B (after) and the change label (0/255).
In this mirror the images are JPEG-compressed (the original release is PNG) and the labels PNG.
The encoded bytes stay in memory (~345 MB for all splits) and are decoded per item.
Tile names are '<split>_<scene>_<k>', k = 0..15 in column-major order over the
1024 px scene (tile k is at row k % 4, column k // 4).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD, LEVIRCD_DIR

_MEAN = np.asarray(IMAGENET_MEAN, np.float32) * 255
_STD = np.asarray(IMAGENET_STD, np.float32) * 255


def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


def load_rows(split: str, root: Path = LEVIRCD_DIR) -> list[tuple[bytes, bytes, bytes]]:
    """(imageA, imageB, label) PNG bytes for one split."""
    import pyarrow.parquet as pq

    path = root / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `satland-download levircd`.")
    table = pq.read_table(path)
    cols = [table.column(c).to_pylist() for c in ("imageA", "imageB", "label")]
    return [(a["bytes"], b["bytes"], m["bytes"]) for a, b, m in zip(*cols, strict=True)]


def load_names(split: str, root: Path = LEVIRCD_DIR) -> list[str]:
    """Tile names such as 'test_100_7' (scene 100, tile 7), aligned with ``load_rows``."""
    import pyarrow.parquet as pq

    col = pq.read_table(root / f"{split}.parquet", columns=["imageA"]).column("imageA")
    return [Path(v["path"]).stem for v in col.to_pylist()]


def decode_row(row: tuple[bytes, bytes, bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (A uint8 HxWx3, B uint8 HxWx3, mask uint8 HxW in {0,1})."""
    a, b, m = (decode_image(x) for x in row)
    if m.ndim == 3:
        m = m[..., 0]
    return a[..., :3], b[..., :3], (m > 127).astype(np.uint8)


def paired_augment(a: np.ndarray, b: np.ndarray, m: np.ndarray, rng: np.random.Generator,
                   swap_prob: float = 0.0):
    """Same flip / 90° rotation for both dates and the mask; optional temporal swap.

    Swapping A and B keeps the label valid for an order-invariant model (|fA - fB|) and teaches
    the network that demolition looks like construction in reverse.
    """
    k = int(rng.integers(0, 4))
    a, b, m = (np.rot90(x, k) for x in (a, b, m))
    if rng.random() < 0.5:
        a, b, m = (x[:, ::-1] for x in (a, b, m))
    if rng.random() < 0.5:
        a, b, m = (x[::-1] for x in (a, b, m))
    if rng.random() < swap_prob:
        a, b = b, a
    return np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(m)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> ImageNet-normalised float 3xHxW."""
    return torch.from_numpy(((img.astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1).copy())


class ChangeDataset(Dataset):
    def __init__(self, rows: list[tuple[bytes, bytes, bytes]], augment: bool = False,
                 seed: int = 0, swap_prob: float = 0.5):
        self.rows = rows
        self.augment = augment
        self.swap_prob = swap_prob
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        a, b, m = decode_row(self.rows[i])
        if self.augment:
            rng = np.random.default_rng((self.seed, self._epoch, i))
            a, b, m = paired_augment(a, b, m, rng, self.swap_prob)
        return to_tensor(a), to_tensor(b), torch.from_numpy(m.astype(np.float32))[None]
