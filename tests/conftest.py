"""Synthetic Sentinel-2 cubes and LEVIR-CD-like tiles: no dataset, GPU or download needed."""

from __future__ import annotations

import io

import numpy as np
import pytest
import tifffile
from PIL import Image

from satland.bands import BAND_NAMES

# Rough reflectance (x10,000) per band for three surface types, in EuroSAT band order.
_SPECTRA = {
    "vegetation": [1200, 900, 800, 500, 1100, 2800, 3300, 3400, 900, 15, 1700, 800, 3600],
    "water": [1300, 1000, 800, 500, 400, 300, 250, 200, 80, 10, 100, 60, 180],
    "urban": [1500, 1400, 1400, 1500, 1600, 1800, 1900, 2000, 700, 20, 2600, 2200, 2100],
}


def synthetic_cube(kind: str = "vegetation", size: int = 16, seed: int = 0) -> np.ndarray:
    """(13, size, size) uint16 reflectance cube with a little noise."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_SPECTRA[kind], np.float32)[:, None, None]
    noise = rng.normal(0, 30, (len(BAND_NAMES), size, size))
    return np.clip(base + noise, 1, 10_000).astype(np.uint16)


@pytest.fixture
def cube():
    return synthetic_cube()


@pytest.fixture
def fake_eurosat(tmp_path):
    """EuroSAT-shaped folder: 4 band-last GeoTIFF-like tifs for each of the 10 classes."""
    from satland.config import EUROSAT_CLASSES

    root = tmp_path / "EuroSAT_MS"
    kinds = list(_SPECTRA)
    for c, cls in enumerate(EUROSAT_CLASSES):
        (root / cls).mkdir(parents=True)
        for i in range(4):
            cube = synthetic_cube(kinds[c % 3], 16, seed=10 * c + i)
            tifffile.imwrite(root / cls / f"{cls}_{i + 1}.tif", np.moveaxis(cube, 0, -1))
    return root


def _encode(arr: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def change_pair(size: int = 64, seed: int = 0):
    """Before/after RGB tiles where a 'building' square appears; returns (a, b, mask01)."""
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(110, 8, (size, size, 3)), 0, 255).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-4, 5, a.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    y, x = rng.integers(4, size // 2, 2)
    mask[y:y + size // 4, x:x + size // 4] = 1
    b[mask == 1] = (210, 90, 70)  # red roof
    return a, b, mask


@pytest.fixture
def fake_levircd(tmp_path):
    """train/val/test parquet files in the Hugging Face mirror's schema (JPEG images, PNG mask)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "levircd"
    root.mkdir()
    for split, n in (("train", 6), ("val", 4), ("test", 16)):
        cols = {"imageA": [], "imageB": [], "label": []}
        for i in range(n):
            a, b, m = change_pair(64, seed=i)
            stem = f"{split}_1_{i}"
            cols["imageA"].append({"bytes": _encode(a, "JPEG"), "path": f"{stem}.jpg"})
            cols["imageB"].append({"bytes": _encode(b, "JPEG"), "path": f"{stem}.jpg"})
            cols["label"].append({"bytes": _encode(m * 255, "PNG"), "path": f"{stem}.png"})
        pq.write_table(pa.table(cols), root / f"{split}.parquet")
    return root
