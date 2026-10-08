import numpy as np
import pytest
import torch

from satland.change import (
    BCEDiceLoss,
    Confusion,
    SiameseUnet,
    change_magnitude,
    load_cd_checkpoint,
    otsu_threshold,
    save_cd_checkpoint,
)
from tests.conftest import change_pair


@pytest.fixture(scope="module")
def siamese():
    torch.manual_seed(0)
    return SiameseUnet("resnet18", pretrained=False).eval()


def test_siamese_is_order_invariant(siamese):
    a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(siamese(a, b), siamese(b, a), atol=1e-5)


def test_identical_dates_give_a_constant_map(siamese):
    a = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = siamese(a, a)
    assert out.shape == (1, 1, 64, 64)
    assert out.std() < 1e-4  # all difference features are zero -> no spatial structure


def test_concat_fusion_shape():
    model = SiameseUnet("resnet18", pretrained=False, fusion="concat").eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)).shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        SiameseUnet("resnet18", pretrained=False, fusion="sum")


def test_loss_prefers_correct_masks():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1
    loss = BCEDiceLoss(0.5, pos_weight=3.0)
    good = loss(torch.where(target > 0, 10.0, -10.0), target)
    bad = loss(torch.where(target > 0, -10.0, 10.0), target)
    assert good < 0.01 < bad


def test_confusion_metrics():
    pred = np.array([[1, 1, 0, 0]], bool)
    target = np.array([[1, 0, 1, 0]], bool)
    c = Confusion().update(pred, target)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.f1() == pytest.approx(0.5) and c.iou() == pytest.approx(1 / 3)
    assert c.precision() == c.recall() == c.oa() == pytest.approx(0.5)
    c.update(pred, target)  # pooling accumulates counts
    assert c.tp == 2 and Confusion().f1() == 0.0


def test_cva_finds_the_new_building():
    a, b, mask = change_pair(64, seed=3)
    mag = change_magnitude(a, b, blur=1)
    assert mag.shape == mask.shape
    pred = mag > otsu_threshold(mag.ravel())
    assert Confusion().update(pred, mask).f1() > 0.8


def test_otsu_splits_bimodal_data():
    rng = np.random.default_rng(0)
    low, high = rng.normal(0, 0.2, 1000), rng.normal(5, 0.2, 1000)
    t = otsu_threshold(np.concatenate([low, high]))
    assert low.max() <= t < high.min()


def test_cd_checkpoint_round_trip(tmp_path, siamese):
    cfg = {"encoder": "resnet18", "fusion": "absdiff"}
    save_cd_checkpoint(siamese, tmp_path / "cd.pt", {"config": cfg, "epoch": 1})
    loaded, meta = load_cd_checkpoint(tmp_path / "cd.pt")
    a, b = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(loaded(a, b), siamese(a, b), atol=1e-6)
    assert meta["epoch"] == 1
