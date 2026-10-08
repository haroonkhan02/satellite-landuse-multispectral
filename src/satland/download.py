"""Download EuroSAT (13-band, Zenodo) and LEVIR-CD (256 px tiles, Hugging Face).

Both sources are open downloads with no login. Downloads resume if interrupted.

Examples:
    satland-download eurosat     # 2.1 GB zip -> data/EuroSAT_MS/<class>/*.tif
    satland-download levircd     # 345 MB of parquet -> data/levircd/{train,val,test}.parquet
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from .config import (
    DATA_DIR,
    EUROSAT_CLASSES,
    EUROSAT_DIR,
    EUROSAT_NUM_IMAGES,
    EUROSAT_URL,
    EUROSAT_ZIP_BYTES,
    LEVIRCD_DIR,
    LEVIRCD_FILES,
    LEVIRCD_HF,
)


def fetch(url: str, dest: Path, expected_bytes: int | None = None, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial file with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected_bytes and have == expected_bytes:
        return dest
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored the range: start over
            have = 0
        total = expected_bytes or (have + int(resp.headers.get("Content-Length", 0)))
        mode = "ab" if have else "wb"
        with open(dest, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                         desc=dest.name) as bar:
            while block := resp.read(chunk):
                f.write(block)
                bar.update(len(block))
    if expected_bytes and dest.stat().st_size != expected_bytes:
        raise OSError(f"{dest.name}: got {dest.stat().st_size} bytes, expected {expected_bytes}"
                      " (re-run to resume)")
    return dest


def count_eurosat(root: Path = EUROSAT_DIR) -> dict[str, int]:
    return {c: len(list((root / c).glob("*.tif"))) for c in EUROSAT_CLASSES}


def download_eurosat(keep_zip: bool = False) -> Path:
    if EUROSAT_DIR.exists() and sum(count_eurosat().values()) == EUROSAT_NUM_IMAGES:
        print(f"EuroSAT already present at {EUROSAT_DIR}")
        return EUROSAT_DIR
    zip_path = fetch(EUROSAT_URL, DATA_DIR / "EuroSAT_MS.zip", EUROSAT_ZIP_BYTES)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".tif")]
        tmp = DATA_DIR / "_eurosat_tmp"
        for m in tqdm(members, desc="extract"):
            zf.extract(m, tmp)
    # The zip nests the class folders one or two levels deep; flatten to data/EuroSAT_MS/<class>.
    first = next(tmp.rglob("*.tif"))
    src_root = first.parent.parent
    EUROSAT_DIR.mkdir(parents=True, exist_ok=True)
    for cls_dir in src_root.iterdir():
        if cls_dir.is_dir():
            shutil.move(str(cls_dir), EUROSAT_DIR / cls_dir.name)
    shutil.rmtree(tmp)
    counts = count_eurosat()
    if sum(counts.values()) != EUROSAT_NUM_IMAGES:
        raise RuntimeError(f"Unexpected EuroSAT contents: {counts}")
    if not keep_zip:
        zip_path.unlink()
    print(f"EuroSAT ready: {sum(counts.values())} patches in {EUROSAT_DIR}")
    return EUROSAT_DIR


def download_levircd() -> Path:
    for split, (name, size, _) in LEVIRCD_FILES.items():
        fetch(f"{LEVIRCD_HF}/{name}", LEVIRCD_DIR / f"{split}.parquet", size)
    print(f"LEVIR-CD tiles ready in {LEVIRCD_DIR}")
    return LEVIRCD_DIR


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=["eurosat", "levircd", "all"])
    ap.add_argument("--keep-zip", action="store_true", help="keep the EuroSAT zip after extraction")
    args = ap.parse_args(argv)
    if args.dataset in ("eurosat", "all"):
        download_eurosat(args.keep_zip)
    if args.dataset in ("levircd", "all"):
        download_levircd()


if __name__ == "__main__":
    main()
