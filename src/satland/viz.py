"""Image helpers: NDVI colouring, labelled tiles and mosaics (pure numpy + Pillow)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .bands import composite, spectral_indices


def ndvi_rgb(cube: np.ndarray) -> np.ndarray:
    """NDVI in [-1, 1] -> brown (bare / water) .. yellow .. green (dense vegetation)."""
    ndvi = spectral_indices(cube)[0]
    t = (np.clip(ndvi, -0.2, 0.8) + 0.2) / 1.0
    stops = np.array([[120, 80, 50], [235, 220, 120], [30, 130, 50]], np.float32)
    lo = np.clip(t * 2, 0, 1)[..., None]
    hi = np.clip(t * 2 - 1, 0, 1)[..., None]
    rgb = np.where(t[..., None] < 0.5, stops[0] + (stops[1] - stops[0]) * lo,
                   stops[1] + (stops[2] - stops[1]) * hi)
    return rgb.astype(np.uint8)


def views(cube: np.ndarray) -> list[np.ndarray]:
    """True colour, colour-infrared, SWIR composite and NDVI for one (13, H, W) patch."""
    return [composite(cube, "rgb"), composite(cube, "cir"), composite(cube, "swir"),
            ndvi_rgb(cube)]


def upscale(img: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def label(img: np.ndarray, text: str, color=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    """Draw a caption bar at the top of an image (after resizing, so text stays legible)."""
    pil = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil)
    w = draw.textlength(text)
    draw.rectangle([0, 0, w + 6, 13], fill=bg)
    draw.text((3, 1), text, fill=color)
    return np.asarray(pil)


def mosaic(rows: list[list[np.ndarray]], pad: int = 4, fill: int = 255) -> np.ndarray:
    """Grid of equally sized HxWx3 tiles with ``pad`` pixels of white between them."""
    h, w = rows[0][0].shape[:2]
    ncol = max(len(r) for r in rows)
    out = np.full((len(rows) * (h + pad) - pad, ncol * (w + pad) - pad, 3), fill, np.uint8)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            out[i * (h + pad):i * (h + pad) + h, j * (w + pad):j * (w + pad) + w] = tile
    return out


def change_overlay(after: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Colour the 'after' image: green = correct change, red = missed, blue = false alarm."""
    out = (after.astype(np.float32) * 0.55).astype(np.uint8)
    pred, target = pred.astype(bool), target.astype(bool)
    out[pred & target] = (40, 200, 70)
    out[~pred & target] = (220, 50, 50)
    out[pred & ~target] = (60, 110, 240)
    return out
