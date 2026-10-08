import numpy as np
import torch

from satland.evaluate_cd import stitch, tile_scores
from satland.levircd import (
    ChangeDataset,
    decode_row,
    load_names,
    load_rows,
    paired_augment,
    to_tensor,
)
from tests.conftest import change_pair


def test_parquet_rows_and_names(fake_levircd):
    rows = load_rows("train", fake_levircd)
    names = load_names("train", fake_levircd)
    assert len(rows) == len(names) == 6 and names[0] == "train_1_0"
    a, b, m = decode_row(rows[0])
    assert a.shape == b.shape == (64, 64, 3) and set(np.unique(m)) <= {0, 1} and m.sum() > 0


def test_paired_augment_moves_images_and_mask_together():
    a, b, m = change_pair(32, seed=1)
    rng = np.random.default_rng(5)
    for _ in range(8):
        a2, b2, m2 = paired_augment(a, b, m, rng, swap_prob=0.0)
        assert np.all(b2[m2 == 1] == (210, 90, 70))  # the roof pixels stay under the mask
        assert m2.sum() == m.sum()


def test_temporal_swap():
    a, b, m = change_pair(32, seed=1)
    a2, b2, _ = paired_augment(a, b, m, np.random.default_rng(0), swap_prob=1.0)
    assert np.any(np.all(a2 == (210, 90, 70), axis=-1))  # the roof is now in the 'before' image


def test_dataset_items(fake_levircd):
    ds = ChangeDataset(load_rows("val", fake_levircd), augment=True)
    a, b, m = ds[0]
    assert a.shape == b.shape == (3, 64, 64) and m.shape == (1, 64, 64)
    assert a.dtype == torch.float32 and abs(float(to_tensor(np.zeros((2, 2, 3), np.uint8)).mean()))
    ds.set_epoch(1)
    assert len(ds) == 4


def test_stitch_row_major_tiles():
    tiles = {k: np.full((2, 2), k) for k in range(16)}
    big = stitch(tiles)
    assert big.shape == (8, 8) and big[0, 6] == 3 and big[6, 0] == 12 and big[7, 7] == 15


def test_tile_scores():
    t = np.zeros((4, 4), bool)
    assert np.isnan(tile_scores(t, t))
    t[0, 0] = True
    assert tile_scores(t, t) == 1.0
