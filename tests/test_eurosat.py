import json

import numpy as np
import torch

from satland.bands import spectral_indices
from satland.eurosat import (
    InputBuilder,
    augment_batch,
    band_stats,
    build_cache,
    iterate_batches,
    list_eurosat,
    load_split,
    read_tif,
    save_split,
    spectral_indices_torch,
    stratified_split,
)


def test_list_and_read_band_last_tifs(fake_eurosat):
    files, labels = list_eurosat(fake_eurosat)
    assert len(files) == 40 and np.bincount(labels).tolist() == [4] * 10
    cube = read_tif(fake_eurosat / files[0])
    assert cube.shape == (13, 16, 16) and cube.dtype == np.uint16


def test_stratified_split_is_disjoint_and_balanced():
    labels = np.repeat(np.arange(10), 100)
    split = stratified_split(labels, (0.7, 0.1, 0.2), seed=0)
    all_idx = np.concatenate(list(split.values()))
    assert len(all_idx) == len(set(all_idx.tolist())) == 1000
    for part, n in (("train", 70), ("val", 10), ("test", 20)):
        assert np.bincount(labels[split[part]]).tolist() == [n] * 10
    again = stratified_split(labels, (0.7, 0.1, 0.2), seed=0)
    assert all(np.array_equal(split[k], again[k]) for k in split)


def test_split_round_trip_by_file_stem(tmp_path, fake_eurosat):
    files, labels = list_eurosat(fake_eurosat)
    split = stratified_split(labels, (0.5, 0.25, 0.25))
    save_split(files, split, tmp_path / "s.json")
    loaded = load_split(files, tmp_path / "s.json")
    assert all(np.array_equal(split[k], loaded[k]) for k in split)


def test_cache_and_band_stats(tmp_path, fake_eurosat):
    cache = build_cache(fake_eurosat, tmp_path / "c.npy", tmp_path / "i.json")
    assert cache.shape == (40, 13, 16, 16)
    meta = json.loads((tmp_path / "i.json").read_text())
    assert len(meta["files"]) == 40
    idx = np.arange(0, 40, 2)
    stats = band_stats(cache, idx, chunk=7)
    ref = np.asarray(cache[idx]).astype(np.float64) / 10_000
    np.testing.assert_allclose(stats["mean"], ref.mean(axis=(0, 2, 3)), rtol=1e-6)
    np.testing.assert_allclose(stats["std"], ref.std(axis=(0, 2, 3)), rtol=1e-4)


def test_input_builder_standardises_selected_bands(cube):
    stats = {"mean": [0.1] * 13, "std": [0.05] * 13}
    builder = InputBuilder([3, 2, 1], with_indices=True, stats=stats)
    raw = torch.from_numpy(cube.astype(np.int32))[None]
    x = builder(raw)
    assert builder.channels == 7 and x.shape == (1, 7, 16, 16)
    expected_red = (cube[3].astype(np.float32) / 10_000 - 0.1) / 0.05
    np.testing.assert_allclose(x[0, 0].numpy(), expected_red, rtol=1e-5)
    np.testing.assert_allclose(x[0, 3:].numpy(), spectral_indices(cube), atol=1e-5)


def test_torch_and_numpy_indices_agree(cube):
    refl = torch.from_numpy(cube.astype(np.float32) / 10_000)[None]
    np.testing.assert_allclose(spectral_indices_torch(refl)[0].numpy(),
                               spectral_indices(cube.astype(np.float32) / 10_000), atol=1e-5)


def test_augment_keeps_pixels_per_sample():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    y = augment_batch(x, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    for i in range(2):
        assert torch.equal(torch.sort(y[i].flatten()).values, torch.sort(x[i].flatten()).values)


def test_iterate_batches_covers_every_index_once():
    cache = np.arange(10 * 13 * 2 * 2, dtype=np.uint16).reshape(10, 13, 2, 2)
    labels = np.arange(10) % 3
    idx = np.array([0, 2, 3, 5, 7, 9])
    seen = []
    for raw, y, b in iterate_batches(cache, labels, idx, 4, shuffle=True,
                                     rng=np.random.default_rng(1)):
        assert raw.shape[1:] == (13, 2, 2) and torch.equal(y, torch.from_numpy(labels[b]))
        seen += b.tolist()
    assert sorted(seen) == idx.tolist()
