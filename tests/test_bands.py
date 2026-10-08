import numpy as np
import pytest

from satland.bands import (
    BAND_NAMES,
    BAND_SETS,
    band_indices,
    composite,
    normalized_difference,
    resolve_band_set,
    spectral_indices,
)
from tests.conftest import synthetic_cube


def test_eurosat_band_order_puts_b8a_last():
    assert len(BAND_NAMES) == 13
    assert BAND_NAMES[7] == "B08" and BAND_NAMES[-1] == "B8A"


def test_resolve_named_and_custom_sets():
    assert resolve_band_set("rgb") == (("B04", "B03", "B02"), False)
    names, with_idx = resolve_band_set("rgb_nir+idx")
    assert names == BAND_SETS["rgb_nir"] and with_idx
    assert resolve_band_set("b04,B08")[0] == ("B04", "B08")
    with pytest.raises(ValueError):
        resolve_band_set("B04,B99")


def test_band_indices_map_to_positions():
    assert band_indices(["B02", "B8A"]) == [1, 12]


def test_normalized_difference_is_bounded_and_safe():
    a = np.array([0.0, 1.0, 0.5])
    b = np.array([0.0, 0.0, 0.5])
    nd = normalized_difference(a, b)
    assert np.all(np.abs(nd) <= 1) and nd[1] == pytest.approx(1, abs=1e-5) and nd[2] == 0


def test_indices_separate_vegetation_water_and_urban():
    veg, water, urban = (spectral_indices(synthetic_cube(k)) for k in
                         ("vegetation", "water", "urban"))
    ndvi, ndwi, ndbi = 0, 1, 2
    assert veg[ndvi].mean() > 0.6 > urban[ndvi].mean()
    assert water[ndwi].mean() > 0 > veg[ndwi].mean()
    assert urban[ndbi].mean() > veg[ndbi].mean()


def test_indices_work_on_batches():
    batch = np.stack([synthetic_cube(seed=i) for i in range(3)])
    out = spectral_indices(batch)
    assert out.shape == (3, 4, 16, 16)
    np.testing.assert_allclose(out[1], spectral_indices(batch[1]), atol=1e-6)


@pytest.mark.parametrize("kind", ["rgb", "cir", "swir"])
def test_composites_are_uint8_images(kind, cube):
    img = composite(cube, kind)
    assert img.shape == (16, 16, 3) and img.dtype == np.uint8


def test_cir_shows_vegetation_red():
    img = composite(np.concatenate([synthetic_cube("vegetation", 8),
                                    synthetic_cube("water", 8)], axis=2), "cir")
    veg, water = img[:, :8].reshape(-1, 3).mean(0), img[:, 8:].reshape(-1, 3).mean(0)
    assert veg[0] > water[0]
