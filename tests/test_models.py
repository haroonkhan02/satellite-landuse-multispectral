import torch

from satland.bands import BAND_NAMES
from satland.models import adapt_stem_weight, build_classifier, load_checkpoint, save_checkpoint


def test_stem_keeps_rgb_filters_and_rescales():
    w = torch.randn(8, 3, 7, 7)
    names = ["B02", "B03", "B04", "B08", "NDVI"]
    new = adapt_stem_weight(w, names)
    scale = 3 / len(names)
    assert new.shape == (8, 5, 7, 7)
    assert torch.allclose(new[:, 2], w[:, 0] * scale)  # B04 = red = pretrained channel 0
    assert torch.allclose(new[:, 0], w[:, 2] * scale)  # B02 = blue
    assert torch.allclose(new[:, 3], w.mean(1) * scale)


def test_classifier_forward_on_13_bands():
    model = build_classifier(list(BAND_NAMES), 10, pretrained=False)
    out = model(torch.randn(2, 13, 64, 64))
    assert out.shape == (2, 10)


def test_checkpoint_round_trip(tmp_path):
    names = ["B04", "B03", "B02", "B08"]
    model = build_classifier(names, 10, pretrained=False).eval()
    meta = {"config": {"arch": "resnet18"}, "channel_names": names, "classes": list("abcdefghij")}
    save_checkpoint(model, tmp_path / "m.pt", meta)
    loaded, ckpt = load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(1, 4, 64, 64)
    assert torch.allclose(model(x), loaded(x)) and ckpt["channel_names"] == names
