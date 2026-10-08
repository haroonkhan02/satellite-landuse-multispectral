"""EuroSAT 13-band data: listing, stratified split, a uint16 memmap cache and batch iteration.

Reading 27,000 GeoTIFFs every epoch is slow, so ``satland-prepare`` decodes them once into a
single (27000, 13, 64, 64) uint16 ``.npy`` file (2.9 GB) that is memory-mapped afterwards.
Batches are built by fancy-indexing the memmap; band selection, normalisation, spectral indices
and flips/rotations run on the training device.

Example:
    satland-prepare              # builds data/eurosat_ms.npy + configs/eurosat_split.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from .bands import BAND_NAMES, NUM_BANDS, REFLECTANCE_SCALE, normalized_difference
from .config import (
    CONFIGS_DIR,
    EUROSAT_CACHE,
    EUROSAT_CLASSES,
    EUROSAT_DIR,
    EUROSAT_INDEX,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
)

SPLIT_FILE = CONFIGS_DIR / "eurosat_split.json"
STATS_FILE = CONFIGS_DIR / "eurosat_band_stats.json"


def list_eurosat(root: Path = EUROSAT_DIR) -> tuple[list[str], np.ndarray]:
    """Sorted '<class>/<file>.tif' paths and their integer labels."""
    files, labels = [], []
    for label, cls in enumerate(EUROSAT_CLASSES):
        names = sorted(p.name for p in (root / cls).glob("*.tif"))
        files += [f"{cls}/{n}" for n in names]
        labels += [label] * len(names)
    if not files:
        raise FileNotFoundError(f"No EuroSAT tifs under {root}; run `satland-download eurosat`.")
    return files, np.asarray(labels, dtype=np.int64)


def read_tif(path: Path) -> np.ndarray:
    """A EuroSAT GeoTIFF as a (13, H, W) uint16 array (files are stored band-last)."""
    arr = tifffile.imread(path)
    if arr.ndim == 3 and arr.shape[-1] == NUM_BANDS:
        arr = np.moveaxis(arr, -1, 0)
    if arr.shape[0] != NUM_BANDS:
        raise ValueError(f"{path}: expected {NUM_BANDS} bands, got shape {arr.shape}")
    return np.ascontiguousarray(arr.astype(np.uint16))


def stratified_split(labels: np.ndarray, fractions=SPLIT_FRACTIONS, seed: int = SPLIT_SEED
                     ) -> dict[str, np.ndarray]:
    """Per-class shuffled split so every class keeps the same proportions in each part."""
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_train = round(fractions[0] * len(idx))
        n_val = round(fractions[1] * len(idx))
        parts["train"] += idx[:n_train].tolist()
        parts["val"] += idx[n_train:n_train + n_val].tolist()
        parts["test"] += idx[n_train + n_val:].tolist()
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()}


def save_split(files: list[str], split: dict[str, np.ndarray], path: Path = SPLIT_FILE) -> None:
    """Stored as file stems (not indices) so it survives a different listing order."""
    data = {k: [Path(files[i]).stem for i in v] for k, v in split.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def load_split(files: list[str], path: Path = SPLIT_FILE) -> dict[str, np.ndarray]:
    stem_to_idx = {Path(f).stem: i for i, f in enumerate(files)}
    data = json.loads(path.read_text())
    return {k: np.sort(np.asarray([stem_to_idx[s] for s in v], dtype=np.int64))
            for k, v in data.items()}


def band_stats(cache: np.ndarray, idx: np.ndarray, chunk: int = 1024) -> dict[str, list[float]]:
    """Per-band mean/std of reflectance (0..1) over the given images (the training split)."""
    s = np.zeros(NUM_BANDS)
    s2 = np.zeros(NUM_BANDS)
    n = 0
    for start in range(0, len(idx), chunk):
        x = cache[np.sort(idx[start:start + chunk])].astype(np.float64) / REFLECTANCE_SCALE
        s += x.sum(axis=(0, 2, 3))
        s2 += (x ** 2).sum(axis=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return {"bands": list(BAND_NAMES), "mean": mean.tolist(), "std": std.tolist()}


def build_cache(root: Path = EUROSAT_DIR, cache_path: Path = EUROSAT_CACHE,
                index_path: Path = EUROSAT_INDEX) -> np.ndarray:
    files, labels = list_eurosat(root)
    first = read_tif(root / files[0])
    out = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint16,
                                    shape=(len(files), *first.shape))
    for i, f in enumerate(tqdm(files, desc="caching tifs")):
        out[i] = read_tif(root / f)
    out.flush()
    index_path.write_text(json.dumps({"files": files, "labels": labels.tolist()}))
    return np.load(cache_path, mmap_mode="r")


def load_cache(cache_path: Path = EUROSAT_CACHE, index_path: Path = EUROSAT_INDEX
               ) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} missing; run `satland-prepare` first.")
    meta = json.loads(index_path.read_text())
    return np.load(cache_path, mmap_mode="r"), meta["files"], np.asarray(meta["labels"])


# ---------------------------------------------------------------------------- model inputs
def spectral_indices_torch(refl: torch.Tensor) -> torch.Tensor:
    """NDVI, NDWI, NDBI, NDMI from a (N, 13, H, W) reflectance tensor (see bands.py)."""
    red, green, nir, swir1 = (refl[:, BAND_NAMES.index(b)] for b in ("B04", "B03", "B08", "B11"))
    idx = torch.stack([normalized_difference(nir, red), normalized_difference(green, nir),
                       normalized_difference(swir1, nir), normalized_difference(nir, swir1)], 1)
    return idx.clamp(-1, 1)


class InputBuilder:
    """Raw uint16 batch (N, 13, H, W) -> standardised selected bands (+ optional indices)."""

    def __init__(self, band_idx: list[int], with_indices: bool, stats: dict, device="cpu"):
        self.band_idx = list(band_idx)
        self.with_indices = with_indices
        mean = torch.tensor(stats["mean"], dtype=torch.float32)[self.band_idx]
        std = torch.tensor(stats["std"], dtype=torch.float32)[self.band_idx]
        self.mean = mean.view(1, -1, 1, 1).to(device)
        self.std = std.view(1, -1, 1, 1).to(device)

    @property
    def channels(self) -> int:
        return len(self.band_idx) + (4 if self.with_indices else 0)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        refl = raw.float() / REFLECTANCE_SCALE
        x = (refl[:, self.band_idx] - self.mean) / self.std
        if self.with_indices:
            x = torch.cat([x, spectral_indices_torch(refl)], dim=1)
        return x


def augment_batch(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random flips and 90-degree rotations per sample: nadir images have no 'up'."""
    n = x.shape[0]
    flips = torch.rand(n, generator=generator) < 0.5
    rots = torch.randint(0, 4, (n,), generator=generator)
    x = x.clone()
    x[flips.to(x.device)] = x[flips.to(x.device)].flip(-1)
    for k in (1, 2, 3):
        sel = (rots == k).to(x.device)
        if sel.any():
            x[sel] = torch.rot90(x[sel], k, dims=(-2, -1))
    return x


def iterate_batches(cache: np.ndarray, labels: np.ndarray, idx: np.ndarray, batch_size: int,
                    shuffle: bool = False, rng: np.random.Generator | None = None
                    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """Yield (raw uint16->int32 tensor, label tensor, image indices); reads are index-sorted."""
    order = idx.copy()
    if shuffle:
        (rng or np.random.default_rng()).shuffle(order)
    for start in range(0, len(order), batch_size):
        b = np.sort(order[start:start + batch_size])
        raw = torch.from_numpy(np.asarray(cache[b]).astype(np.int32))
        yield raw, torch.from_numpy(labels[b]), b


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the EuroSAT memmap cache, split and stats.")
    ap.add_argument("--root", type=Path, default=EUROSAT_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild the cache even if it exists")
    args = ap.parse_args(argv)
    if EUROSAT_CACHE.exists() and not args.force:
        cache, files, labels = load_cache()
    else:
        cache = build_cache(args.root)
        cache, files, labels = load_cache()
    if SPLIT_FILE.exists():
        split = load_split(files)
        print(f"Using committed split {SPLIT_FILE.name}")
    else:
        split = stratified_split(labels)
        save_split(files, split)
        print(f"Wrote {SPLIT_FILE}")
    if not STATS_FILE.exists() or args.force:
        STATS_FILE.write_text(json.dumps(band_stats(cache, split["train"]), indent=1))
        print(f"Wrote {STATS_FILE}")
    print({k: len(v) for k, v in split.items()}, "cache", cache.shape, cache.dtype)


if __name__ == "__main__":
    main()
