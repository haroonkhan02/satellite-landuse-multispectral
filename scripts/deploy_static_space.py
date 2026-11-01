"""Publish the in-browser demo: weights and ONNX files to a model repo, web/ to a static Space.

Uses the Hugging Face login stored by `hf auth login`. The page loads the INT8 ONNX files from
the model repo, so the Space only holds HTML, JavaScript and a few test examples:
EuroSAT patches as raw uint16 cubes (13 x 64 x 64, band-first) and LEVIR-CD tile pairs with
their change masks. Examples are picked by rule from the test splits with the INT8 models.

Example:
    python scripts/deploy_static_space.py --dry-run   # build outputs/static_space only
    python scripts/deploy_static_space.py --models    # also upload weights and ONNX files
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from satland.config import EUROSAT_CLASSES
from satland.eurosat import load_cache, load_split
from satland.export import OnnxRunner
from satland.export_models import EXPORT_DIR, cd_batches, cls_batches
from satland.levircd import decode_row, load_names, load_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "haroonx02/satellite-landuse-multispectral"
SPACE_REPO = "haroonx02/satellite-landuse-browser"
WEB_FILES = ("index.html", "app.js", "satland.js", "README.md")
HUB_FILES = {f"onnx/{n}": f"models/onnx/{n}" for n in (
    "eurosat_all13_int8.onnx", "eurosat_all13_fp16.onnx", "eurosat_rgb_int8.onnx",
    "levircd_siamese_int8.onnx", "levircd_siamese_fp16.onnx")}
HUB_FILES.update({"eurosat_all13.pt": "models/eurosat_all13.pt",
                  "eurosat_rgb.pt": "models/eurosat_rgb.pt",
                  "levircd_absdiff.pt": "models/levircd_absdiff.pt",
                  "README.md": "hub/MODEL_CARD_HUB.md"})


def pick_eurosat(extra: int = 2) -> list[dict]:
    """One test patch per class that both INT8 models get right, then patches the RGB model
    gets wrong and the 13-band model gets right (different true classes)."""
    cache, files, labels = load_cache()
    idx = np.sort(load_split(files)["test"])
    preds = {}
    for name in ("all13", "rgb"):
        run = OnnxRunner(EXPORT_DIR / f"eurosat_{name}_int8.onnx")
        preds[name] = np.concatenate([run.run(f) for f in cls_batches(cache, idx)]).argmax(1)
    y = labels[idx]
    both = (preds["all13"] == y) & (preds["rgb"] == y)
    fixed = (preds["all13"] == y) & (preds["rgb"] != y)
    chosen = [(int(np.flatnonzero(both & (y == c))[0]), "") for c in range(len(EUROSAT_CLASSES))]
    seen = set()
    for j in np.flatnonzero(fixed):
        if y[j] not in seen and len(seen) < extra:
            seen.add(y[j])
            wrong = EUROSAT_CLASSES[preds["rgb"][j]]
            chosen.append((int(j), f"RGB model says {wrong}"))
    out = []
    for j, note in chosen:
        stem = Path(files[idx[j]]).stem
        out.append({"file": f"{stem}.bin", "label": EUROSAT_CLASSES[y[j]], "note": note,
                    "cube": np.asarray(cache[idx[j]])})
    return out


def pick_levir() -> list[dict]:
    """Test tiles: a large and a small new-building case the model gets right, a tile with no
    change and no false alarm, and the worst-scoring tile with real change (honest failure)."""
    rows, names = load_rows("test"), load_names("test")
    run = OnnxRunner(EXPORT_DIR / "levircd_siamese_int8.onnx")
    probs = np.concatenate([run.run(f)[:, 0] for f in cd_batches(rows)])
    stats = []
    for i, r in enumerate(rows):
        t = decode_row(r)[2].astype(bool)
        p = probs[i] > 0.5
        denom = p.sum() + t.sum()
        stats.append((t.mean(), 2 * (p & t).sum() / denom if denom else 1.0, p.sum()))
    share, f1, npred = (np.asarray(v) for v in zip(*stats, strict=True))
    big = np.flatnonzero((share > 0.2) & (f1 > 0.9))
    small = np.flatnonzero((share > 0.02) & (share < 0.06) & (f1 > 0.85))
    empty = np.flatnonzero((share == 0) & (npred == 0))
    hard = np.flatnonzero(share > 0.05)
    picks = {"new_estate": big[0], "few_houses": small[0], "no_change": empty[0],
             "hard_case": hard[np.argmin(f1[hard])]}
    return [{"name": f"{k + 1}_{key}", "row": rows[i], "tile": names[i], "f1": float(f1[i]),
             "share": float(share[i])} for k, (key, i) in enumerate(picks.items())]


def build_bundle(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    ex = out / "examples"
    ex.mkdir(parents=True)
    for name in WEB_FILES:
        shutil.copy(ROOT / "web" / name, out / name)
    manifest = {"eurosat": [], "levir": [], "levir_info": []}
    for item in pick_eurosat():
        item.pop("cube").astype("<u2").tofile(ex / item["file"])
        manifest["eurosat"].append(item)
    for item in pick_levir():
        row = item.pop("row")  # the mirror's own JPEG bytes, not re-encoded
        (ex / f"{item['name']}_before.jpg").write_bytes(row[0])
        (ex / f"{item['name']}_after.jpg").write_bytes(row[1])
        Image.fromarray(decode_row(row)[2] * 255).save(ex / f"{item['name']}_label.png")
        manifest["levir"].append(item["name"])
        manifest["levir_info"].append(item)
    (ex / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=SPACE_REPO)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "static_space")
    parser.add_argument("--models", action="store_true", help="upload weights + ONNX files too")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(args.out)
    print(f"Bundle ready at {bundle}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    if args.models:
        api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
        for dest, src in HUB_FILES.items():
            api.upload_file(path_or_fileobj=ROOT / src, path_in_repo=dest, repo_id=MODEL_REPO,
                            commit_message=f"Add {Path(dest).name}")
    api.create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=bundle,
                      commit_message="Deploy browser demo")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
