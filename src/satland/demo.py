"""Gradio demo: land-use classification of a 13-band patch and before/after change detection.

Example:
    satland-demo            # http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import EUROSAT_CLASSES, MODELS_DIR
from .eurosat import STATS_FILE, InputBuilder, read_tif
from .levircd import to_tensor
from .models import load_checkpoint, pick_device
from .viz import upscale, views


class Classifier:
    def __init__(self, ckpt: Path, device):
        self.model, meta = load_checkpoint(ckpt, device)
        self.builder = InputBuilder(meta["band_idx"], meta["with_indices"],
                                    json.loads(STATS_FILE.read_text()), device)
        self.device = device

    @torch.no_grad()
    def __call__(self, cube: np.ndarray) -> dict[str, float]:
        raw = torch.from_numpy(cube.astype(np.int32))[None].to(self.device)
        probs = torch.softmax(self.model(self.builder(raw)), 1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(EUROSAT_CLASSES, probs, strict=True)}


@torch.no_grad()
def detect_change(model, before: np.ndarray, after: np.ndarray, device,
                  threshold: float = 0.5) -> np.ndarray:
    """Full-image inference; pads to a multiple of 32 (the encoder's total stride)."""
    h, w = before.shape[:2]
    a, b = to_tensor(before)[None].to(device), to_tensor(after)[None].to(device)
    ph, pw = (-h) % 32, (-w) % 32
    a, b = F.pad(a, (0, pw, 0, ph), mode="reflect"), F.pad(b, (0, pw, 0, ph), mode="reflect")
    prob = torch.sigmoid(model(a, b))[0, 0, :h, :w].cpu().numpy()
    return prob > threshold


def build_app(device):
    import gradio as gr

    from .change import load_cd_checkpoint

    classifiers = {name: Classifier(MODELS_DIR / f"eurosat_{name}.pt", device)
                   for name in ("all13", "rgb") if (MODELS_DIR / f"eurosat_{name}.pt").exists()}
    cd_path = MODELS_DIR / "levircd_absdiff.pt"
    cd_model = load_cd_checkpoint(cd_path, device)[0] if cd_path.exists() else None

    def classify(file):
        if file is None:
            return [None] * 4 + [{}, {}]
        cube = read_tif(Path(file))
        imgs = [upscale(v, 256) for v in views(cube)]
        outs = [classifiers[k](cube) if k in classifiers else {} for k in ("all13", "rgb")]
        return imgs + outs

    def change(before, after, threshold):
        if cd_model is None or before is None or after is None:
            return None
        if before.shape != after.shape:
            raise gr.Error("Before and after images must have the same size.")
        pred = detect_change(cd_model, before[..., :3], after[..., :3], device, threshold)
        out = (after[..., :3] * 0.55).astype(np.uint8)
        out[pred] = (40, 200, 70)
        return out

    with gr.Blocks(title="Satellite land use & change") as app:
        gr.Markdown("# Sentinel-2 land use (EuroSAT) · building change (LEVIR-CD)")
        with gr.Tab("Land-use classification"):
            gr.Markdown("Upload a 13-band EuroSAT GeoTIFF (e.g. from `data/EuroSAT_MS/`).")
            f = gr.File(label="EuroSAT .tif", file_types=[".tif"])
            with gr.Row():
                ims = [gr.Image(label=t) for t in ("true colour", "colour infrared", "SWIR",
                                                   "NDVI")]
            with gr.Row():
                lab13 = gr.Label(num_top_classes=3, label="13-band model")
                labrgb = gr.Label(num_top_classes=3, label="RGB-only model")
            f.change(classify, f, [*ims, lab13, labrgb])
        with gr.Tab("Change detection"):
            gr.Markdown("Two co-registered RGB images of the same place (0.5 m works best). "
                        "Green = detected building change.")
            with gr.Row():
                before = gr.Image(label="before", type="numpy")
                after = gr.Image(label="after", type="numpy")
            thr = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="threshold")
            out = gr.Image(label="change")
            gr.Button("Detect change").click(change, [before, after, thr], out)
    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    build_app(pick_device(args.device)).launch(server_port=args.port)


if __name__ == "__main__":
    main()
