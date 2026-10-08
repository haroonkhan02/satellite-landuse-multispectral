"""Paths, dataset constants and the YAML-backed training configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = ROOT / "outputs"

# EuroSAT (Helber et al., 2019): 27,000 64x64 Sentinel-2 patches, 10 land-use classes, 13 bands.
EUROSAT_URL = "https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content"
EUROSAT_ZIP_BYTES = 2_065_402_329
EUROSAT_DIR = DATA_DIR / "EuroSAT_MS"
EUROSAT_CACHE = DATA_DIR / "eurosat_ms.npy"  # (27000, 13, 64, 64) uint16 memmap
EUROSAT_INDEX = DATA_DIR / "eurosat_index.json"  # file order + labels of the cache
EUROSAT_NUM_IMAGES = 27_000
EUROSAT_CLASSES: tuple[str, ...] = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)

# LEVIR-CD (Chen & Shi, 2020): 637 1024x1024 bi-temporal Google Earth pairs, 0.5 m, building
# change masks. The Hugging Face mirror below holds the official split cut into 256x256 tiles.
LEVIRCD_HF = "https://huggingface.co/datasets/ericyu/LEVIRCD_Cropped256/resolve/main/data"
LEVIRCD_FILES = {
    "train": ("train-00000-of-00001-737f96f51caac8cd.parquet", 241_029_927, 7120),
    "val": ("val-00000-of-00001-d09d88a7419f2427.parquet", 34_557_458, 1024),
    "test": ("test-00000-of-00001-31d7c3e3444e5b5d.parquet", 69_534_024, 2048),
}
LEVIRCD_DIR = DATA_DIR / "levircd"

SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.7, 0.1, 0.2)  # train / val / test, stratified per class
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _YamlConfig:
    """Shared YAML loading with unknown-key checking."""

    @classmethod
    def load(cls, path: str | Path | None = None):
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **kwargs):
        """Return a copy with the non-None keyword values replaced (for CLI flags)."""
        data = self.to_dict()
        data.update({k: v for k, v in kwargs.items() if v is not None})
        return type(self)(**data)


@dataclass
class ClsConfig(_YamlConfig):
    """EuroSAT classification run; saved into the checkpoint and the history."""

    bands: str = "all13"  # a key of bands.BAND_SETS, optionally with "+idx"
    arch: str = "resnet18"
    pretrained: bool = True
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    workers: int = 0  # the cache is a memmap; worker processes add more overhead than they save
    seed: int = 0
    train_fraction: float = 1.0  # stratified subset of the training split (low-label study)


@dataclass
class CDConfig(_YamlConfig):
    """LEVIR-CD change-detection run."""

    encoder: str = "resnet18"
    pretrained: bool = True
    fusion: str = "absdiff"  # "absdiff" (Siamese, |fA - fB|) or "concat" (early fusion, 6 ch)
    epochs: int = 15
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    pos_weight: float = 3.0  # changed pixels are ~5 % of the data
    workers: int = 4
    seed: int = 0
    max_train: int = 0  # 0 = all 7,120 training tiles
