import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")

from satland import export  # noqa: E402
from satland.bands import BAND_NAMES, band_indices  # noqa: E402
from satland.change import SiameseUnet, save_cd_checkpoint  # noqa: E402
from satland.config import EUROSAT_CLASSES  # noqa: E402
from satland.eurosat import InputBuilder  # noqa: E402
from satland.models import build_classifier, save_checkpoint  # noqa: E402

from .conftest import change_pair, synthetic_cube  # noqa: E402

STATS = {"mean": np.linspace(0.05, 0.25, 13).tolist(), "std": np.linspace(0.02, 0.1, 13).tolist()}


def _cls_checkpoint(path, bands=BAND_NAMES):
    torch.manual_seed(0)
    model = build_classifier(list(bands), len(EUROSAT_CLASSES), pretrained=False).eval()
    meta = {"config": {"arch": "resnet18"}, "channel_names": list(bands),
            "band_idx": band_indices(bands), "with_indices": False, "stats": STATS,
            "classes": list(EUROSAT_CLASSES)}
    save_checkpoint(model, path, meta)
    return path


@pytest.fixture(scope="module")
def cls_ckpt(tmp_path_factory):
    return _cls_checkpoint(tmp_path_factory.mktemp("cls") / "all13.pt")


@pytest.fixture(scope="module")
def cd_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    model = SiameseUnet("resnet18", pretrained=False).eval()
    path = tmp_path_factory.mktemp("cd") / "cd.pt"
    save_cd_checkpoint(model, path, {"config": {"encoder": "resnet18", "fusion": "absdiff"}})
    return path


@pytest.fixture(scope="module")
def cls_onnx(cls_ckpt, tmp_path_factory):
    wrapper, _ = export.load_classifier(cls_ckpt)
    return export.export_onnx(wrapper, "cls", tmp_path_factory.mktemp("onnx") / "cls.onnx")


@pytest.fixture(scope="module")
def cd_onnx(cd_ckpt, tmp_path_factory):
    return export.export_onnx(export.load_change(cd_ckpt), "cd",
                              tmp_path_factory.mktemp("onnx") / "cd.onnx")


def _patches(n=3):
    kinds = ("vegetation", "water", "urban")
    return np.stack([synthetic_cube(kinds[i % 3], 64, seed=i) for i in range(n)]).astype(np.float32)


def _tiles(n=2):
    pairs = [change_pair(256, seed=i) for i in range(n)]
    a, b = (np.stack([p[k] for p in pairs]).transpose(0, 3, 1, 2).astype(np.float32) / 255
            for k in (0, 1))
    return {"before": a, "after": b}


@pytest.mark.parametrize("bands", [BAND_NAMES, ("B04", "B03", "B02")])
def test_classifier_wrapper_matches_training_preprocessing(tmp_path, bands):
    wrapper, meta = export.load_classifier(_cls_checkpoint(tmp_path / "m.pt", bands))
    raw = torch.from_numpy(_patches())
    builder = InputBuilder(meta["band_idx"], False, STATS)
    with torch.no_grad():
        ref = torch.softmax(wrapper.model(builder(raw)), 1)
        out = wrapper(raw)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out.sum(1), torch.ones(len(raw)))


def test_classifier_onnx_parity_and_dynamic_batch(cls_onnx, cls_ckpt):
    wrapper, _ = export.load_classifier(cls_ckpt)
    x = _patches(5)
    ref = export.TorchRunner(wrapper).run({"bands": x})
    out = export.OnnxRunner(cls_onnx).run({"bands": x})
    assert out.shape == (5, 10)
    assert np.abs(out - ref).max() < 1e-5
    one = export.OnnxRunner(cls_onnx).run({"bands": x[:1]})
    np.testing.assert_allclose(one, out[:1], atol=1e-6)


def test_change_onnx_parity_and_order_invariance(cd_onnx, cd_ckpt):
    feed = _tiles()
    ref = export.TorchRunner(export.load_change(cd_ckpt)).run(feed)
    run = export.OnnxRunner(cd_onnx)
    out = run.run(feed)
    assert out.shape == (2, 1, 256, 256)
    assert np.abs(out - ref).max() < 1e-4
    swapped = run.run({"before": feed["after"], "after": feed["before"]})
    np.testing.assert_allclose(swapped, out, atol=1e-5)


def test_fp16_halves_the_file_and_stays_close(cls_onnx, tmp_path):
    fp16 = export.to_fp16(cls_onnx, tmp_path / "cls16.onnx")
    assert fp16.stat().st_size < 0.6 * cls_onnx.stat().st_size
    x = {"bands": _patches()}
    a = export.OnnxRunner(cls_onnx).run(x)
    b = export.OnnxRunner(fp16).run(x)
    assert b.dtype == np.float32 and np.abs(a - b).max() < 0.02


def test_int8_quarter_size_float_tail(cd_onnx, tmp_path):
    import onnx

    calib = [_tiles(1) for _ in range(3)]
    int8 = export.to_int8(cd_onnx, tmp_path / "cd8.onnx", calib)
    assert int8.stat().st_size < 0.35 * cd_onnx.stat().st_size
    nodes = onnx.load(str(int8)).graph.node
    assert nodes[-1].op_type == "Sigmoid"  # applied to the dequantised float logits
    assert nodes[-2].op_type == "DequantizeLinear"
    out = export.OnnxRunner(int8).run(_tiles())
    assert out.shape == (2, 1, 256, 256) and out.min() >= 0 and out.max() <= 1


def test_run_batches_respects_max_batch(cls_onnx):
    run = export.OnnxRunner(cls_onnx)
    run.max_batch = 2
    x = _patches(5)
    out, ms = run.run_batches([{"bands": x}])
    assert out.shape == (5, 10) and ms > 0
    np.testing.assert_allclose(out, run.run({"bands": x}), atol=1e-6)


def test_index_feature_classifiers_are_rejected(tmp_path):
    path = _cls_checkpoint(tmp_path / "m.pt")
    ckpt = torch.load(path, weights_only=False)
    ckpt["with_indices"] = True
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="band-only"):
        export.load_classifier(path)
