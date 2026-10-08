"""Sentinel-2 band metadata, band-set presets and spectral indices.

EuroSAT stores all 13 Sentinel-2 L1C bands as B01..B12 followed by B8A (the narrow-NIR band
comes *last*, not after B08; the per-band means give it away: B8A ~ B08 ~ 0.23 while the
water-vapour band B09 is ~0.07 and the cirrus band B10 ~0.001). Values are top-of-atmosphere
reflectance scaled by 10,000 (so ~0..10000, stored as uint16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    name: str
    description: str
    wavelength_nm: int  # central wavelength (Sentinel-2A)
    resolution_m: int  # native ground sampling distance


BANDS: tuple[Band, ...] = (
    Band("B01", "coastal aerosol", 443, 60),
    Band("B02", "blue", 490, 10),
    Band("B03", "green", 560, 10),
    Band("B04", "red", 665, 10),
    Band("B05", "red edge 1", 705, 20),
    Band("B06", "red edge 2", 740, 20),
    Band("B07", "red edge 3", 783, 20),
    Band("B08", "NIR", 842, 10),
    Band("B09", "water vapour", 945, 60),
    Band("B10", "SWIR cirrus", 1375, 60),
    Band("B11", "SWIR 1", 1610, 20),
    Band("B12", "SWIR 2", 2190, 20),
    Band("B8A", "narrow NIR", 865, 20),
)
BAND_NAMES: tuple[str, ...] = tuple(b.name for b in BANDS)
NUM_BANDS = len(BANDS)
REFLECTANCE_SCALE = 10_000.0

# Band subsets compared in the ablation study (names, not indices, so they read well in reports).
BAND_SETS: dict[str, tuple[str, ...]] = {
    "rgb": ("B04", "B03", "B02"),
    "rgb_nir": ("B04", "B03", "B02", "B08"),
    "rgb_nir_swir": ("B04", "B03", "B02", "B08", "B11", "B12"),
    "no_rgb": ("B01", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"),
    "all13": BAND_NAMES,
}

# Indices appended as extra input channels when a band set name ends in "+idx".
INDEX_NAMES: tuple[str, ...] = ("NDVI", "NDWI", "NDBI", "NDMI")

# Pretrained ImageNet weights expect R, G, B in this order.
RGB_BANDS: tuple[str, ...] = ("B04", "B03", "B02")


def band_indices(names: tuple[str, ...] | list[str]) -> list[int]:
    """Positions of the named bands in the 13-band EuroSAT cube."""
    unknown = [n for n in names if n not in BAND_NAMES]
    if unknown:
        raise ValueError(f"Unknown Sentinel-2 bands: {unknown}")
    return [BAND_NAMES.index(n) for n in names]


def resolve_band_set(spec: str) -> tuple[tuple[str, ...], bool]:
    """'rgb_nir+idx' -> (('B04','B03','B02','B08'), True). Also accepts 'B04,B08,B11'."""
    with_idx = spec.endswith("+idx")
    base = spec[:-4] if with_idx else spec
    if base in BAND_SETS:
        names = BAND_SETS[base]
    else:
        names = tuple(s.strip().upper() for s in base.split(",") if s.strip())
        band_indices(names)  # validates
    if not names:
        raise ValueError(f"Empty band set: {spec!r}")
    return names, with_idx


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(a - b) / (a + b), safe for zeros; works on numpy arrays or torch tensors."""
    return (a - b) / (a + b + eps)


def spectral_indices(cube: np.ndarray) -> np.ndarray:
    """Compute NDVI, NDWI, NDBI, NDMI from a (13, H, W) or (N, 13, H, W) reflectance cube.

    - NDVI = (NIR - Red) / (NIR + Red)        vegetation vigour (chlorophyll absorbs red)
    - NDWI = (Green - NIR) / (Green + NIR)    open water (McFeeters 1996)
    - NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)    built-up / bare soil
    - NDMI = (NIR - SWIR1) / (NIR + SWIR1)    vegetation water content
    Returns an array with the band axis replaced by the 4 indices, each in [-1, 1].
    """
    x = cube.astype(np.float32)
    axis = x.ndim - 3

    def band(name: str) -> np.ndarray:
        return np.take(x, BAND_NAMES.index(name), axis=axis)

    red, green, nir, swir1 = band("B04"), band("B03"), band("B08"), band("B11")
    out = [
        normalized_difference(nir, red),
        normalized_difference(green, nir),
        normalized_difference(swir1, nir),
        normalized_difference(nir, swir1),
    ]
    return np.clip(np.stack(out, axis=axis), -1.0, 1.0)


def _stretch(img: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def composite(cube: np.ndarray, kind: str = "rgb") -> np.ndarray:
    """(13, H, W) reflectance -> (H, W, 3) uint8 display image.

    rgb = true colour (B04, B03, B02); cir = colour infrared (B08, B04, B03), vegetation shows red;
    swir = (B12, B08, B04), separates bare soil, urban and water.
    """
    combos = {"rgb": ("B04", "B03", "B02"), "cir": ("B08", "B04", "B03"),
              "swir": ("B12", "B08", "B04")}
    if kind not in combos:
        raise ValueError(f"Unknown composite {kind!r}; choose from {sorted(combos)}")
    rgb = np.stack([cube[BAND_NAMES.index(n)] for n in combos[kind]], axis=-1).astype(np.float32)
    return (_stretch(rgb) * 255).astype(np.uint8)
